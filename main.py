import argparse
import functools
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import neptune
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_checkpoint
import torch.nn as nn
import torch.nn.functional as F
import torch.version
from datasets import load_dataset, load_from_disk
from datasets.distributed import split_dataset_by_node
from neptune.utils import stringify_unsupported
from neptune_pytorch import NeptuneLogger
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    transformer_auto_wrap_policy,
    wrap,
)
from torch.nn.attention import SDPBackend
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import GPT2TokenizerFast

# setup below main()
global_rank = 0
local_rank = 0
world_size = 1


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = (
            torch.arange(seq_len, dtype=torch.long, device=x.device)
            .unsqueeze(0)
            .expand_as(x)
        )
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        embeddings = token_embeddings + position_embeddings
        return embeddings


class AttentionLayer(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super(AttentionLayer, self).__init__()

        self.ln = nn.LayerNorm(dmodel)

        self.heads = heads

        self.input_projection = nn.Linear(dmodel, 3 * dmodel, bias=False)

        self.output_projection = nn.Linear(dmodel, dmodel, bias=False)

    def forward(self, x, attention_mask):
        x = self.ln(x)

        projected = self.input_projection(x)

        batch, seq_len = x.shape[:-1]
        q_chunk, k_chunk, v_chunk = torch.chunk(projected, chunks=3, dim=-1)
        query = q_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        key = k_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        value = v_chunk.view(batch, seq_len, self.heads, -1).transpose(1, 2)

        with torch.nn.attention.sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        ):
            attention_output = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attention_mask,
                is_causal=True,
            )

        output = self.output_projection(attention_output.transpose(1, 2).flatten(-2))

        return output


def FeedForward(
    dmodel,
):
    return nn.Sequential(
        OrderedDict(
            [
                ("ff_layernorm", nn.LayerNorm(dmodel)),
                (
                    "pre_relu",
                    nn.Linear(
                        dmodel,
                        4 * dmodel,
                        bias=True,
                    ),
                ),
                ("relu", nn.ReLU()),
                (
                    "post_relu",
                    nn.Linear(
                        4 * dmodel,
                        dmodel,
                        bias=True,
                    ),
                ),
            ]
        )
    )


class Block(nn.Module):
    def __init__(
        self,
        dmodel,
        heads,
    ):
        super().__init__()
        self.attention_layer = AttentionLayer(dmodel, heads)
        self.feed_forward_layer = FeedForward(dmodel)

    def forward(self, x, attention_mask):
        out_attention = self.attention_layer(x, attention_mask)
        x = x + out_attention

        out_feed_forward = self.feed_forward_layer(x)
        x = x + out_feed_forward
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_layer = EmbeddingLayer(
            config.vocab_size, config.d_model, config.max_len
        )
        self.blocks = nn.ModuleList(
            [Block(config.d_model, config.num_heads) for _ in range(config.num_layers)]
        )

        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        output = self.embedding_layer(input_ids)

        for block in self.blocks:
            output = block(output, attention_mask)

        output = self.head(output)
        return output


def create_sched(optimizer, config):
    warmup_steps = round(config.train_steps / 100)
    linear = LinearLR(optimizer=optimizer, total_iters=warmup_steps)
    # cosine = CosineAnnealingWarmRestarts(
    #    optimizer=optimizer, T_0=10, T_mult=2, eta_min=config.learning_rate / 1000
    # )
    # I think this is what the extrapolation paper suggested
    cosine = CosineAnnealingLR(
        optimizer=optimizer, T_max=config.train_steps - warmup_steps
    )
    return SequentialLR(
        optimizer=optimizer, schedulers=[linear, cosine], milestones=[warmup_steps]
    )


def collate_tokenize(tokenizer, sequence_length, data):
    text_batch = [element["text"] for element in data]
    tokenized = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=sequence_length + 1,
    )
    input_ids = tokenized["input_ids"]
    tokenized["input_ids"] = input_ids[:, :-1]
    tokenized["target_ids"] = input_ids[:, 1:]
    tokenized["attention_mask"] = tokenized["attention_mask"][:, :-1]
    return tokenized


def get_dataloader(
    config,
    split="train",
    buffer_size=10000,
    seed=42,
    num_workers=2,
):
    batch_size = config.batch_size
    sequence_length = config.seq_length
    if split == "train":
        if config.is_plgrid:
            hf_dataset = load_from_disk(
                "/net/tscratch/people/plgkciebiera/datasets/c4/train"
            )
        else:
            hf_dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    else:
        if config.is_plgrid:
            hf_dataset = load_from_disk(
                "/net/tscratch/people/plgkciebiera/datasets/c4/validation"
            )
        else:
            hf_dataset = load_dataset(
                "allenai/c4", "en", split="validation", streaming=True
            )
    if config.is_plgrid:
        hf_dataset = hf_dataset.to_iterable_dataset(num_shards=64)
    hf_dataset = hf_dataset.shuffle(buffer_size=buffer_size, seed=seed)
    hf_dataset = split_dataset_by_node(
        hf_dataset, world_size=world_size, rank=global_rank
    )
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    # sampler = DistributedSampler(hf_dataset) if config.is_dist else None
    dataloader = DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_tokenize, tokenizer, sequence_length),
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        # sampler=sampler,
    )
    return dataloader


def calculate_valid_loss(model, valid_dataloader, device, validation_steps):
    valid_losses = []
    for _, batch in zip(range(validation_steps), valid_dataloader):
        with torch.no_grad():
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch["attention_mask"]
            outputs = model(input_ids)
            mask_loss = F.cross_entropy(
                outputs.flatten(0, -2),
                target_ids.reshape(-1).long(),
                reduction="none",
            )
            mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
            loss = mask_loss.mean().item()
            valid_losses.append(loss)
    mean_valid_loss = sum(valid_losses) / validation_steps
    return mean_valid_loss


def train_model(config, device):
    dataloader = get_dataloader(config)
    valid_dataloader = get_dataloader(config, split="validation")
    validation_steps = int(
        1e06 // (config.batch_size * config.seq_length)
    )  # we want to evaluate on 1M tokens
    # conditions and some stuff below adapted from here
    # https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and int(torch.version.cuda.split(".")[0]) >= 11
        and dist.is_nccl_available()
        and torch.cuda.nccl.version() >= (2, 10)
    )

    avail_precision = MixedPrecision(
        param_dtype=torch.bfloat16 if bf16_ready else torch.float16,
        reduce_dtype=torch.bfloat16 if bf16_ready else torch.float16,
    )
    my_trans_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block,
        },
    )
    model1 = Transformer(config)
    torch.cuda.set_device(local_rank)
    model1.to(device)
    model1 = FSDP(
        model1, auto_wrap_policy=my_trans_wrap_policy, mixed_precision=avail_precision
    )

    if config.log and global_rank == 0:
        nep_run = neptune.init_run(
            name=f"{os.environ['SLURM_JOB_NAME']}-{os.environ['SLURM_JOB_ID']}",
            tags=os.environ["SLURM_NODELIST"],
        )
        nep_log = NeptuneLogger(
            run=nep_run,
            model=model1,
        )
        nep_run[nep_log.base_namespace]["hyperparams"] = stringify_unsupported(
            config.__dict__
        )
    if global_rank == 0:
        print(f"Are we using bf16?: {bf16_ready}")
        if config.log:
            nep_run[nep_log.base_namespace]["hyperparams/bf16_used"] = bf16_ready

    # cosine annealing without warm restarts
    optimizer = AdamW(model1.parameters(), lr=config.learning_rate)
    scheduler = create_sched(optimizer, config)
    steps_done = 0
    if config.load_dir is not None:
        model1, optimizer, scheduler, steps_done = load_model(
            model1, optimizer, scheduler, steps_done, config.load_dir
        )

    model1.train()

    for i, batch in zip(range(config.train_steps), dataloader):
        # I don't think there's a more efficent way to do this with huggingface
        if i < steps_done:
            if i % 50 == 0:
                print(f"skipping step {i}")
            if config.log and global_rank == 0 and i % config.log_train_loss_freq == 0:
                nep_run[nep_log.base_namespace]["train/loss"].append(0.0)
                nep_run[nep_log.base_namespace]["train/lr"].append(0.0)
            continue
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        attention_mask = batch["attention_mask"]

        optimizer.zero_grad()
        outputs = model1(input_ids)

        mask_loss = F.cross_entropy(
            outputs.flatten(0, -2),
            target_ids.reshape(-1).long(),
            reduction="none",
        )
        mask_loss = mask_loss[attention_mask.reshape(-1) == 1]
        loss = mask_loss.mean()

        if i % config.log_train_loss_freq == 0:
            train_loss = torch.zeros(1)
            train_loss += loss.item()
            dist.reduce(train_loss, 0)
            if global_rank == 0:
                train_loss /= world_size
                train_loss = train_loss.item()
                print(
                    f"Step:{i}, Train Loss:{train_loss}, Learn Rate:{scheduler.get_last_lr()[0]}"
                )
                if config.log:
                    nep_run[nep_log.base_namespace]["train/loss"].append(train_loss)
                    nep_run[nep_log.base_namespace]["train/lr"].append(
                        scheduler.get_last_lr()[0]
                    )

        # this was originally config.log_train_loss_freq so I changed it
        if i % config.log_valid_loss_freq == 0:
            valid_loss = torch.zeros(1)
            valid_loss += calculate_valid_loss(
                model1, valid_dataloader, device, validation_steps
            )
            dist.reduce(valid_loss, 0)
            if global_rank == 0:
                valid_loss /= world_size
                valid_loss = valid_loss.item()
                print(f"Valid loss: {valid_loss}")
                if config.log:
                    nep_run[nep_log.base_namespace]["valid/loss"].append(valid_loss)

        loss.backward()
        optimizer.step()
        scheduler.step()
        if config.early_stop > 0 and i == config.early_stop - 1:
            if config.log and global_rank == 0:
                for j in range(config.early_stop, config.train_steps):
                    if j % config.log_train_loss_freq == 0:
                        nep_run[nep_log.base_namespace]["train/loss"].append(0.0)
                        nep_run[nep_log.base_namespace]["train/lr"].append(0.0)
            break

    print("training done")
    valid_loss = torch.zeros(1)
    valid_loss += calculate_valid_loss(
        model1, valid_dataloader, device, validation_steps
    )
    dist.reduce(valid_loss, 0)
    if global_rank == 0:
        valid_loss /= world_size
        valid_loss = valid_loss.item()
        if config.log:
            nep_run[nep_log.base_namespace]["valid/loss-final"].append(valid_loss)
        print(f"Final valid loss:{valid_loss}")
    if config.save_dir is not None:
        save_model(model1, optimizer, scheduler, i + 1, config.save_dir)
    if global_rank == 0 and config.log:
        nep_run.stop()


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(f"Device type is: {device}. Remember to train on GPU.")
        exit(-1)
    train_model(config, device)


# ------------------------- Setup -----------------------------


def save_model(model, optim, sched, step, dir):
    model_sd, optim_sd = get_state_dict(model, optim)
    state_dict = {
        "model": model_sd,
        "optim": optim_sd,
        "sched": sched.state_dict(),
        "steps_done": step,
    }
    dist_checkpoint.save(
        state_dict=state_dict, storage_writer=dist_checkpoint.FileSystemWriter(dir)
    )


def load_model(model, optim, sched, step, dir):
    model_sd, optim_sd = get_state_dict(model, optim)
    sched_sd = sched.state_dict()
    state_dict = {
        "model": model_sd,
        "optim": optim_sd,
        "sched": sched_sd,
        "steps_done": step,
    }
    dist_checkpoint.load(
        state_dict=state_dict, storage_reader=dist_checkpoint.FileSystemReader(dir)
    )
    set_state_dict(
        model=model,
        optimizers=optim,
        model_state_dict=model_sd,
        optim_state_dict=optim_sd,
    )
    sched.load_state_dict(sched_sd)
    return (
        model,
        optim,
        sched,
        state_dict["steps_done"],
    )


def is_pos_int(x):
    xc = int(x)
    if xc <= 0:
        raise argparse.ArgumentTypeError(f"{x} is not a positive integer")
    return xc


def is_nonneg_float(x):
    xc = float(x)
    if xc < 0:
        raise argparse.ArgumentTypeError(f"{x} is not a non negative float")
    return xc


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bml-2", description="Second BML project network"
    )
    parser.add_argument(
        "-n",
        "--train_steps",
        type=is_pos_int,
        default=1000,
        help="number of training steps",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default="50257",
        help="Number of unique tokens in input dict",
    )
    parser.add_argument(
        "--max_len",
        type=is_pos_int,
        default=256,
        help="Max number of tokens on input for embedding layer",
    )
    parser.add_argument(
        "--d_model", type=is_pos_int, default=256, help="Model hidden size"
    )
    parser.add_argument(
        "--num_heads", type=is_pos_int, default=4, help="Number of heads in MHA"
    )
    parser.add_argument(
        "--num_layers", type=is_pos_int, default=4, help="Number of transformer layers "
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=is_nonneg_float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=is_nonneg_float,
        default=0.0,
        help="Dropout rate",
    )
    parser.add_argument(
        "--seq_length",
        type=is_pos_int,
        default=256,
        help="Sequence length for dataloader",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=is_pos_int,
        default=64,
        help="Batch size use by dataloader per gpu",
    )
    parser.add_argument(
        "--log_train_loss_freq",
        type=is_pos_int,
        default=100,
        help="How often to record loss during training",
    )
    parser.add_argument(
        "--log_valid_loss_freq",
        type=is_pos_int,
        default=100,
        help="How often to record loss during validation",
    )

    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument(
        "--is_dist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to set up distributed params",
    )
    parser.add_argument(
        "--is_plgrid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether this is run on plgrid (loading K. Ciebera dataset)",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="dir to save model to, empty if we shouldn't save",
    )
    parser.add_argument(
        "--load_dir",
        default=None,
        help="dir to load model from, empty if we don't want to load",
    )
    parser.add_argument(
        "--early_stop",
        default=-1,
        type=int,
        help="Used for the model saving and loading task",
    )
    return parser


def setup(config):
    global global_rank
    global local_rank
    global world_size

    if not config.is_dist:
        if os.environ["MASTER_ADDR"] is None:
            os.environ["MASTER_ADDR"] = "localhost"
        if os.environ["MASTER_PORT"] is None:
            os.environ["MASTER_PORT"] = "12355"
    else:
        assert os.environ["RANK"] is not None
        assert os.environ["LOCAL_RANK"] is not None
        assert os.environ["WORLD_SIZE"] is not None
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"my global rank is {global_rank}")
        print(f"my local rank is {local_rank}")
        print(f"world size is {world_size}")
        print(f"device count is {torch.cuda.device_count()}")

    # initialize the process group
    # dist.init_process_group(
    #    "cpu:gloo,cuda:nccl", rank=global_rank, world_size=world_size
    # )
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=global_rank, world_size=world_size
    )


def cleanup():
    print("destroying process group")
    dist.destroy_process_group()
    print("process group destroyed")


if __name__ == "__main__":
    parser = create_parser()
    config = parser.parse_args()
    if config.max_len < config.seq_length:
        raise argparse.ArgumentTypeError("max_len shorter than seq_length")
    if config.load_dir is not None:
        load_dir = Path(config.load_dir)
        config.load_dir = load_dir
        if not load_dir.is_dir():
            raise argparse.ArgumentTypeError(f"Invalid load path: {str(load_dir)}")
    if config.save_dir is not None:
        save_dir = Path(config.save_dir)
        config.save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        if len(os.listdir(save_dir)) > 0:
            raise argparse.ArgumentTypeError(f"Save dir not empty: {str(save_dir)}")
    print("args parsed")
    setup(config)
    print("setup-complete")
    main(config)
    cleanup()
