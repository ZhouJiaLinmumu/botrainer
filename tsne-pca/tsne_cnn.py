from __future__ import print_function
import argparse
import math
import multiprocessing
import numpy as np
import os
from pandas import DataFrame
import pickle
import sys
sys.path.append("..")
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

 from cnn import Net
import cnn_dataset
import cnn_setting as setting
import file
import plot
import tsne_config
import tsne_dataset
import util


 # Add command line parser
parser = argparse.ArgumentParser(description="Visualization for CNN model")
parser.add_argument('-lr', '--learning_rate',
                    help='learning rate', default=setting.lr, type=float)
parser.add_argument('-m', '--momentum', help='learning rate',
                    default=setting.momentum, type=float)
parser.add_argument('-b', '--batch', help='training batch size',
                    default=setting.batch_size, type=int)
parser.add_argument('-tb', '--test_batch', help='test batch size',
                    default=setting.test_batch_size, type=int)
parser.add_argument('-e', '--epoch', help='epochs of training',
                    default=setting.epochs, type=int)
parser.add_argument('-c', '--cuda', help="Whether use cuda",
                    default=setting.use_cuda, type=int)
parser.add_argument('-w', '--way', help='the way of dimension reduction, \
                    "n" for selected features layers, \
                    "m" for cnn middle layers', default="n", type=str)

 paras = parser.parse_args()
# Prepare the dataset
device = torch.device("cuda" if paras.cuda else "cpu")
# train_dataset = tsne_dataset.RemoteMyDataset(cnn_dataset.train_file_id, host="http://localhost")
# test_dataset = tsne_dataset.RemoteMyDataset(cnn_dataset.test_file_id, host="http://localhost")

 train_dataset = tsne_dataset.CSVMyDataset(tsne_config.train_file_id)
test_dataset = tsne_dataset.CSVMyDataset(tsne_config.test_file_id)

 train_loader = DataLoader(train_dataset, batch_size=paras.batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=paras.test_batch, shuffle=False)

 class InterceptionNet(Net):
    def __init__(self):
        super(InterceptionNet, self).__init__()

     def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x_middle = x.view(-1, self.view_size)
        x = F.relu(self.fc1(x_middle))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), x_middle


 feature_data_tsne = None
label_data = None
inference_data = None
label_img = None


 def label_encoding(item):
    ret = ""
    for (key, value) in item.iteritems():
        ret += key
        ret += "="
        ret += str(value)
        ret += "&"
    return ret

 def preload():
    print("Start preload the static dataset.....")
    global feature_data_tsne, label_data, label_img, inference_data

     feature_data = []
    label_data = []

     for data, infos in test_loader:
        feature_data.append(data)
        encoding_labels = DataFrame(infos).apply(
            label_encoding, axis=1).tolist()
        label_data.extend(encoding_labels)
        print ("{0} data is finished".format(str(len(label_data))))
    feature_data = torch.cat(feature_data)

     label_img = feature_data
    feature_data_tsne = feature_data.view(feature_data.size()[0], -1)
    print("Finished preload the static dataset.....")

 def normal_feature_visualization():
    global feature_data_tsne, label_data, label_img
    writer = SummaryWriter(
        comment="Label & Inference Visualization", log_dir="normal_run")
    import pdb; pdb.set_trace()
    writer.add_embedding(feature_data_tsne,
        metadata=label_data, label_img=label_img, global_step=0)

 def middle_layer_feature_visualization():
    global label_data, label_img
    writer = SummaryWriter(
        comment='middle layer of cnn', log_dir="middle_run")

     model = InterceptionNet().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=paras.learning_rate, momentum=paras.momentum)

     for epoch in range(1, paras.epoch + 1):
        # Training procedure
        for data, infos in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            output, middle = model(data)
            target = torch.LongTensor(infos["tag"].tolist()).to(device)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

         # Inference procedure
        with torch.no_grad():
            confusion = np.array([[0, 0], [0, 0]])
            middle_data = []
            predict_label = []

             sum_loss = 0
            sum_correct = 0

             for data, infos in test_loader:
                data, target = data.to(device), target.to(device)
                output, middle = model(data)
                middle = middle.view(data.size()[0], -1)
                middle_data.append(middle)

                 infos = DataFrame(infos)
                target = torch.LongTensor(infos["tag"].tolist()).to(device)
                loss = F.nll_loss(output, target, reduction='sum').item()
                sum_loss += loss
                pred = output.max(1, keepdim=True)[1]
                predict_label.extend(infos.apply(
                    label_encoding, axis=1).tolist())

                 correct = pred.eq(target.view_as(pred)).sum().item()
                sum_correct += correct

             middle_data = torch.cat(middle_data)
            middle_data_tsne = middle_data.view(middle_data.size()[0], -1)
            writer.add_embedding(middle_data_tsne,
                                    metadata=label_data, label_img=label_img, global_step=epoch)

             # Calculate confusion matrix
            guess = pred.cpu().numpy()
            tag = target.view_as(pred).cpu().numpy()
            for i in range(0, len(tag)):
                confusion[tag[i], guess[i]] += 1

             sum_loss /= len(test_loader.dataset)

             print('\nTest set: Loss: %.3f, Accuracy: %d/%d (%.1f%%), Precision: %.1f%%, Recall: %.1f%%\n' % (
                sum_loss, sum_correct, len(test_loader.dataset),
                100. * sum_correct / len(test_loader.dataset),
                100. * confusion[1, 1] / (confusion[1, 1] + confusion[0, 1]),
                100. * confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])
            ))


 def selector(mode):
    if mode == "n":
        ## Normal mode
        normal_feature_visualization()
    if mode == "m":
        # CNN middle layer mode
        middle_layer_feature_visualization()


 if __name__ == "__main__":
    preload()
    selector(paras.way)