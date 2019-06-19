import csv
import ipaddress
import os
import pandas as pd
import sys
sys.path.append("..")
import traceback
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader, TensorDataset

import config
from lstm_encode import *
import util

def read_session_tags(path):
    res = []
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            res.append(int(row[1]))
    return res


def padding_func(batch):
    # Let's assume that each element in "batch" is a tuple (data, label).
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Also need to store the length of each sequence
    # This is later needed in order to unpad the sequences
    lengths = torch.LongTensor([len(x) for x in sequences])
    # Don't forget to grab the labels of the *sorted* batch
    labels = torch.LongTensor(list(map(lambda x: x[1], sorted_batch)))
    return sequences_padded, lengths, labels
 

class LSTMDataset(Dataset):

    def __init__(self, file_id, label=False, row_encoder=encode_row):
        self.label = label
        self.guess_list = None
        self.row_encoder = row_encoder
        self.dataset = self.parse(file_id)
            

    def __getitem__(self, index):
        return self.dataset[index], (int(self.guess_list[index]) if self.guess_list else -1)


    def __len__(self):
        return len(self.dataset)

    def parse(self, file_id):
        print ("LLL")
        if self.label:
            self.guess_list = read_session_tags(util.get_session_tag_path(file_id))
        
        lis = []
        stream_dir = config.STREAM_DIR + file_id
        count = -1
        for filename in os.listdir(stream_dir):
            count += 1
            if count % 1000 == 0:
                print ("{0} files have already loaded".format(str(count)))

            
            raw_data = pd.read_csv(\
            os.path.join(stream_dir, filename), \
                    header=None)
            raw_data.columns = ["time", "id", "web", "type", 'b1', 'b2', 'b3', \
                'b4', 'b5', '5', '6', '7', '8', 'ip', '9', '10', 'status', 'agent', \
                'a', 'b', 'c', 'd', 'e', 'f', 'host', 'query', 'i', 'j', 'k', 'l', 'm', 'n', 'o']

            encode_data = []
            complete = True
            for _, row_raw_data in raw_data.iterrows():
                try:
                    encode_data_row = self.row_encoder(row_raw_data)
                except ipaddress.AddressValueError as e:
                    traceback.print_stack()
                    self.guess_list[count] = -2 ##empty
                    complete = False
                    break
                encode_data.append(encode_data_row)
            if complete is True:
                lis.append(torch.Tensor(encode_data))

                
        self.guess_list = list(filter(lambda x: x != -2, self.guess_list))
        import pdb; pdb.set_trace()
        return lis


if __name__ == "__main__":
    dataset = LSTMDataset("bing-en-us/logs_20190105_www.bing.com_hour_0_refined", label=False)