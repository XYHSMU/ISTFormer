import util
import argparse
import torch
from ISTFormer import ISTFormer
import numpy as np
import pandas as pd

import xyh_util

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="number of input_dim")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="'/Ablation/epoch100_20240326_fulltrees_delete_Conv3_and_Conv4'PEMS08/best_model.pth",
    help="",
)
args = parser.parse_args()


def main():

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        num_nodes = 170
        embeding_dim = 152
        tool = 'xyh_util'

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        num_nodes = 358
        args.es_patience = 100
        embeding_dim = 152
        tool = 'xyh_util'


    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        num_nodes = 307
        embeding_dim = 152
        tool = 'xyh_util'


    elif args.data == "PEMS07":
        args.data = "data//" + args.data
        num_nodes = 883
        embeding_dim = 152
        tool = 'xyh_util'


    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        num_nodes = 250
        embeding_dim = 152
        tool = 'util'


    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        num_nodes = 250
        embeding_dim = 152
        tool = 'util'

    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        num_nodes = 266
        embeding_dim = 152
        tool = 'util'


    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        num_nodes = 266
        embeding_dim = 152
        tool = 'util'

    device = torch.device(args.device)

    model = ISTFormer(
        device, args.input_dim, num_nodes, args.dropout,embeding_dim
    )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print("model load successfully")


    if tool == 'xyh_util':
        dataloader = xyh_util.load_data(args.data, args.batch_size)
        pass
    else:
        dataloader = util.load_dataset(
            args.data, args.batch_size, args.batch_size, args.batch_size
        )
        pass

    scaler = dataloader["scaler"]
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    awmape = []
    armse = []


    for i in range(12):
        if tool == 'xyh_util':
            pred = scaler.inverse_normal(yhat[:,:, i])
            pass
        else:
            pred = scaler.inverse_transform(yhat[:, :, i])
            pass

        real = realy[:,:, i]
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(awmape)))

if __name__ == "__main__":
    main()
