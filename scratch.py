from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ExponentialLR
from torch.optim import AdamW
import torch
from pathlib import Path

def main():
    #model = torch.nn.Linear(10,10)
    #optim = AdamW(model.parameters(), lr=1.0)
    ##sched = ExponentialLR(optim, start_factor=1/3, end_factor=1, total_iters=5)
    #sched = ExponentialLR(optim, start_factor=1/3, end_factor=1, total_iters=5)
    #for i in range(5):
    #    sched.step()
    #    print(sched.get_last_lr())
    for i in range(10):
        continue
    print(i)


if __name__ == "__main__":
    main()
