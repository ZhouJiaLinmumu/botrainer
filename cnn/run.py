from __future__ import print_function
import os
import pickle
from tensorboardX import SummaryWriter
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import numpy as np

from cnn_model import *
import cnn_dataset
import cnn_setting as setting
import file
import metadata
import plot
import util

all_losses = []

def train(model, device, train_loader, optimizer, epoch, sum_start):
    model.train()
    start = time.time()
    for batch_i, (data, target, index) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if setting.verbose:
            print('(%.0f%%) Epoch: %d, Batch: %d, Batch size: %d, Loss: %.3f' % (
                100. * batch_i / len(train_loader),
                epoch, batch_i, target.shape[0], loss.item()
            ))
        all_losses.append(loss.item())

    end = time.time()
    if setting.verbose:
        print('')
    print('Train set: Epoch: %d, Batch size: %d, Loss: %.3f, Dataset size: %d, Throughput: %.3f, Elapsed time: %.4fs, Total time: %.4fs' % (
        epoch, train_loader.batch_size, loss.item(),
        len(train_loader.dataset), len(train_loader.dataset) / (end - start),
        end - start,
        end - sum_start
    ))
    if setting.verbose:
        print('')

    # plot.plot_loss(all_losses)


def test(model, device, test_loader, train_file_id, test_file_id):
    confusion = [[0, 0], [0, 0]]
    confusion_req = [[0, 0], [0, 0]]
    guesses = []

    model.eval()
    sum_loss = 0
    sum_correct = 0
    sum_correct_req = 0
    with torch.no_grad():
        sum_start = time.time()
        batch_i = 0
        for data, target, index in test_loader:
            start = time.time()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction='sum').item()
            sum_loss += loss  # sum up batch loss

            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            guess = pred.view_as(target).cpu().numpy()
            tag = target.view_as(target).cpu().numpy()

            guess_list = []
            correct = 0
            correct_req = 0
            for i in range(0, len(tag)):
                count = test_loader.dataset.counts[int(index.cpu().numpy()[i])]
                confusion[tag[i]][guess[i]] += 1
                confusion_req[tag[i]][guess[i]] += count

                guess_list.append(int(guess[i]))

                if tag[i] == guess[i]:
                    correct += 1
                    correct_req += count

            guesses += guess_list
            sum_correct += correct
            sum_correct_req += correct_req

            end = time.time()
            if setting.verbose:
                print('Batch: %d, Batch size: %d, Test loss: %.3f, Accuracy: %d/%d (%.1f%%), Elapsed time: %.4fs, Latency (per session): %.4fs' % (
                    batch_i, target.shape[0],
                    loss / target.shape[0], correct, target.shape[0],
                    100. * correct / target.shape[0],
                    end - start,
                    (end - start) / target.shape[0]
                ))
            batch_i += 1

    sum_loss /= len(test_loader.dataset)
    sum_end = time.time()

    # plot.plot_matrix(setting.n_categories, confusion)
    # plot.plot_matrix(setting.n_categories, confusion_req)
    file.write_session_guess_file(util.get_session_guess_path(test_file_id), guesses)

    if setting.verbose:
        print('')
    print('Test set: Loss: %.3f, Accuracy: %d/%d (%.1f%%), Precision: %.1f%%, Recall: %.1f%%, Total time: %.4fs, Latency (per session): %.4fs' % (
        sum_loss, sum_correct, len(test_loader.dataset),
        100. * sum_correct / len(test_loader.dataset),
        0 if confusion[1][1] + confusion[0][1] == 0 else 100. * confusion[1][1] / (confusion[1][1] + confusion[0][1]),
        0 if confusion[1][1] + confusion[1][0] == 0 else 100. * confusion[1][1] / (confusion[1][1] + confusion[1][0]),
        sum_end - sum_start,
        (sum_end - sum_start) / len(test_loader.dataset)
    ))
    print('Accuracy (req): %.1f%%, Precision (req): %.1f%%, Recall (req): %.1f%%' % (
        100. * sum_correct_req / (confusion_req[0][0] + confusion_req[0][1] + confusion_req[1][0] + confusion_req[1][1]),
        0 if confusion_req[1][1] + confusion_req[0][1] == 0 else 100. * confusion_req[1][1] / (confusion_req[1][1] + confusion_req[0][1]),
        0 if confusion_req[1][1] + confusion_req[0][1] == 0 else 100. * confusion_req[1][1] / (confusion_req[1][1] + confusion_req[1][0]),
    ))
    if setting.verbose:
        print('')

    metadata.write_result(train_file_id, test_file_id, confusion, confusion_req)


def run(train_file_id, test_file_id):
    torch.manual_seed(setting.seed)

    device = torch.device("cuda" if setting.use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if setting.use_cuda else {}
    train_loader = DataLoader(cnn_dataset.MyDataset(train_file_id),
                              batch_size=setting.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(cnn_dataset.MyDataset(test_file_id),
                             batch_size=setting.test_batch_size, shuffle=False, **kwargs)

    # model = AlexNet().to(device)
    # model = resnet34().to(device)
    model = squeezenet1_0().to(device)

    # util.kill_tensorboard()
    # util.clear_tmp_dir()
    # util.start_tensorboard()
    # writer = SummaryWriter()
    # writer.add_graph(Net(), (torch.rand(setting.batch_size, 1, setting.image_size, setting.image_size),))

    if setting.use_local_model:
        cnn_path = 'cnn.pt'
        loss_path = 'loss.pt'
    else:
        # cnn_path = util.get_model_path(train_file_id)
        cnn_path = "cnn/model.pt"
        loss_path = util.get_loss_path(train_file_id)

    if setting.load_model and os.path.isfile(cnn_path):
        if setting.use_cuda:
            model.load_state_dict(torch.load(cnn_path))
        else:
            model.load_state_dict(torch.load(cnn_path, map_location='cpu'))
        model.eval()
        if os.path.isfile(loss_path):
            with open(loss_path, 'rb') as f:
                global all_losses
                all_losses = pickle.load(f)

    optimizer = optim.SGD(model.parameters(), lr=setting.lr, momentum=setting.momentum)

    if not setting.inference_only:
        sum_start = time.time()
        for epoch in range(1, setting.epochs + 1):
            train(model, device, train_loader, optimizer, epoch, sum_start)
            test(model, device, test_loader, train_file_id, test_file_id)

            if setting.save_model:
                torch.save(model.state_dict(), "cnn/model_lenet.pt")
                with open(loss_path, 'wb') as f:
                    pickle.dump(all_losses, f)
    else:
        test(model, device, test_loader, train_file_id, test_file_id)


if __name__ == '__main__':
    run(cnn_dataset.train_file_id, cnn_dataset.test_file_id)
