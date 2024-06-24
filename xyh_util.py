import numpy as np
import os
import scipy.sparse as sp
import torch
import sys

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):


        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
                # print(self.current_ind)

        return _wrapper()

class XYHScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        pass

    def normal(self, data):
        return (data - self.mean) / self.std

    def inverse_normal(self, data):
        return data * self.std + self.mean

def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def load_data(dataset,batch_size):
    data={}
    url = os.path.join(dataset,os.path.basename(dataset)+".npz")
    print(url)
    origin_data = np.load(url)['data'][...,:1]
    TE = np.zeros([origin_data.shape[0], 2])
    TE[:, 0] = np.array([i%288/288 for i in range(origin_data.shape[0])])
    TE[:, 1] = np.array([int(i // 288) % 7 for i in range(origin_data.shape[0])])
    TE_expanded = np.repeat(np.expand_dims(TE, axis=1), repeats=origin_data.shape[1], axis=1)
    merged_data = np.concatenate((origin_data, TE_expanded), axis=2)

    train_num = round(0.6 * origin_data.shape[0])
    test_num = round(0.2 * origin_data.shape[0])
    val_num = origin_data.shape[0] - train_num - test_num

    trainData,valData,testData = merged_data[:train_num,:,:], merged_data[train_num:train_num+val_num,:,:],merged_data[-test_num:,:,:]


    trainX, trainY = seq2instance(trainData, 12, 12)
    valX, valY = seq2instance(valData, 12, 12)
    testX, testY = seq2instance(testData, 12, 12)

    data['x_train'] = trainX
    data['y_train'] = trainY
    data['x_val'] = valX
    data['y_val'] = valY
    data['x_test'] = testX
    data['y_test'] = testY

    scaler = XYHScaler(mean=data['x_train'][...,0].mean(), std=data['x_train'][...,0].std())
    for category in ["train", "val", "test"]:
        data["x_" + category][..., 0] = scaler.normal(data["x_" + category][..., 0])
        pass

    print("scaler:mean:{},std:{}".format(scaler.mean, scaler.std))
    print('x_train.shape:{},x_val.shape:{},x_test.shape:{}'.format(trainX.shape, valX.shape, testX.shape))
    print('y_train.shape:{},y_val.shape:{},y_test.shape:{}'.format(trainY.shape, valY.shape, testY.shape))


    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))
    random_train = torch.randperm(random_train.size(0))
    data["x_train"] = data["x_train"][random_train, ...]
    data["y_train"] = data["y_train"][random_train, ...]


    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]


    data['scaler'] = scaler
    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], batch_size)

    print()
    return data




if __name__ == '__main__':
    load_data(dataset='data\PEMS08',batch_size=64)
    print('hello')

