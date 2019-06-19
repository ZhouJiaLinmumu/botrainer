import argparse
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import math
import numpy as np

import lstm_encode
from lstm_dataset import *
from lstm_model import PADRNN
import session
import lstm_setting
import util


# Add command line parser
parser = argparse.ArgumentParser(description="LSTM Training Model")
parser.add_argument('-lr', '--learning_rate',
                    help='learning rate', default=lstm_setting.lr, type=float)
parser.add_argument('-b', '--train_batch', help='training batch size',
                    default=lstm_setting.train_batch_size, type=int)
parser.add_argument('-tb', '--test_batch', help='testing batch size',
                    default=lstm_setting.test_batch_size, type=int)
parser.add_argument('-d', '--train_dataset', help='training dataset path',
                    default=lstm_setting.train_dataset_id, type=str)
parser.add_argument('-td', '--test_dataset', help='testing dataset path',
                    default=lstm_setting.test_dataset_id, type=str)
parser.add_argument('-e', '--epoch', help='epochs of training',
                    default=lstm_setting.epochs, type=int)
parser.add_argument('-c', '--cuda', help="use cuda",
                    default=lstm_setting.use_cuda and torch.cuda.is_available(), type=int)
parser.add_argument('-m', '--mode', help="inference(i) or training(t)",
                    default=lstm_setting.mode, type=str)
parser.add_argument('-hl', '--n_hidden', help="hidden_size",
                    default=lstm_setting.lstm_hidden_size, type=int)
parser.add_argument('-hs', '--n_layers', help="layer size",
                    default=lstm_setting.lstm_layers, type=int)
parser.add_argument('-l', "--load", help='Whether load data',
                    default=False, type=int)
paras = parser.parse_args()
device = torch.device("cuda" if paras.cuda else "cpu")

def train(dataset, model, writer, save=True):
    bot_count = 0
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=paras.learning_rate)
    all_losses = []
    dataloader = DataLoader(dataset=dataset,\
                        batch_size=paras.train_batch,\
                            collate_fn=padding_func)

    for iter in range(paras.epoch):
        tp_count = 0
        tn_count = 0 
        fp_count = 0 
        fn_count = 0
        total_loss = 0
        for feature_tensor, x_lengths, label_tensor in dataloader:
            feature_tensor = feature_tensor.to(device)
            x_lengths = x_lengths.to(device)
            label_tensor = label_tensor.to(device)

            output = model(feature_tensor, x_lengths)
            loss = loss_function(output, label_tensor)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
            _, predict = torch.max(output, 1)
            tn, fp, fn, tp = confusion_matrix(label_tensor.cpu().numpy(), predict.cpu().numpy(), labels=[0,1]).ravel()
            tp_count += tp
            tn_count += tn

            fn_count += fn
            fp_count += fp
        
        if iter % 100 == 0:
            print ("Epoch {0}:".format(str(iter)))
            print ("TP {0}, TN{1}, FN{2}, FP{3}".format(str(tp_count), str(tn_count), str(fn_count), str(fp_count)))
            print ("Loss is {0}".format(str(total_loss)))
            print ()

        if writer:
            writer.add_scalar('Loss', total_loss, iter)
        if save:
            torch.save(model.state_dict(), './model.pt')

        iter += 1

def test(dataset, model):
    dataloader = DataLoader(dataset=dataset,\
                        batch_size=paras.train_batch,\
                            collate_fn=padding_func)

    acc = None
    count = 0 
    print (len(dataset))
    tp_count = 0
    tn_count = 0 
    fp_count = 0 
    fn_count = 0
    for feature_tensor, x_lengths, label_tensor in dataloader:
        feature_tensor = feature_tensor.to(device)
        x_lengths = x_lengths.to(device)
        label_tensor = label_tensor.to(device)

        output = model(feature_tensor, x_lengths)
        _, predict = torch.max(output, 1)

        tn, fp, fn, tp = confusion_matrix(label_tensor.cpu().numpy(), predict.cpu().numpy(), labels=[0,1]).ravel()
        tp_count += tp
        tn_count += tn
        fn_count += fn
        fp_count += fp
    import pdb; pdb.set_trace()
    print ("TP {0}, FP{1}, FN{2}, FP{3}".format(str(tp_count), str(tn_count), str(fn_count), str(fp_count)))

def main():
    writer = SummaryWriter()
    model = PADRNN(lstm_encode.encoding_vector_size, paras.n_hidden, 2, paras.n_layers)
    model = model.to(device)
    ## Train
    if paras.mode == "t":
        if paras.mode == "l":
            model.load_state_dict(torch.load('./model.pt'))
        train_dataset = LSTMDataset(paras.train_dataset, label=True)
        train(train_dataset,model, writer, save=True)

    ## Inference
    if paras.mode == "i":
        model.load_state_dict(torch.load('./model.pt'))
        test_dataset = LSTMDataset(paras.test_dataset, label=True)
        test(test_dataset, model)

if __name__ == '__main__':
    main()