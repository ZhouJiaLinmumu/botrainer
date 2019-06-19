import sys
sys.path.append("..")
import config

lr = 0.1

train_batch_size = 10000
test_batch_size = 1000

## May be published later
train_dataset_id = "xxx"
test_dataset_id = "xxx"
lstm_hidden_size = 4
lstm_layers = 1


mode = "t" ##train
epochs = 1000
use_cuda = True