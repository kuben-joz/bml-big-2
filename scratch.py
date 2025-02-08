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

Step:0, Train Loss:11.1875, Learn Rate:3.3333333333333335e-05
Valid loss: 11.202868461608887
Step:100, Train Loss:7.6875, Learn Rate:9.797464868072493e-05
Valid loss: 7.779201030731201
Step:200, Train Loss:7.5625, Learn Rate:9.118382907149173e-05
Valid loss: 7.543032646179199
Step:300, Train Loss:7.40625, Learn Rate:8.028048435688347e-05
Valid loss: 7.410860538482666
Step:400, Train Loss:7.28125, Learn Rate:6.635339816587124e-05
Valid loss: 7.31301212310791
Step:500, Train Loss:7.28125, Learn Rate:5.0793298191740526e-05
Valid loss: 7.238729476928711
Final valid loss:7.238217213114754

Step:600, Train Loss:7.78125, Learn Rate:9.801910851476529e-05
Valid loss: 7.778176307678223
Step:700, Train Loss:7.46875, Learn Rate:9.127359484813878e-05
Valid loss: 7.551741600036621
Step:800, Train Loss:7.53125, Learn Rate:8.040659226635104e-05
Valid loss: 7.411373138427734
Step:900, Train Loss:7.21875, Learn Rate:6.65032553542318e-05
Valid loss: 7.309938430786133
Final valid loss:7.243340163934426


Step:0, Train Loss:11.1875, Learn Rate:3.3333333333333335e-05
Valid loss: 11.1875
Step:100, Train Loss:7.65625, Learn Rate:9.797464868072493e-05
Valid loss: 7.778688430786133
Step:200, Train Loss:7.5625, Learn Rate:9.118382907149173e-05
Valid loss: 7.548155784606934
Step:300, Train Loss:7.4375, Learn Rate:8.028048435688347e-05
Valid loss: 7.418032646179199
Step:400, Train Loss:7.3125, Learn Rate:6.635339816587124e-05
Valid loss: 7.321208953857422
Step:500, Train Loss:7.28125, Learn Rate:5.0793298191740526e-05
Valid loss: 7.24948787689209
Step:600, Train Loss:7.1875, Learn Rate:3.5153981233586354e-05
Valid loss: 7.202356338500977
Step:700, Train Loss:7.09375, Learn Rate:2.099715452144016e-05
Valid loss: 7.172643661499023
Step:800, Train Loss:7.25, Learn Rate:9.736487123447095e-06
Valid loss: 7.153176307678223
Step:900, Train Loss:7.0625, Learn Rate:2.4964441129527396e-06
Valid loss: 7.147541046142578
Final valid loss:7.14702868852459
