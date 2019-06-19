import torch

image_size = 256
n_categories = 2
batch_size = 64
test_batch_size = 64
epochs = 100
lr = 0.01
momentum = 0.5
no_cuda = False
seed = 1

save_model = False
load_model = False
inference_only = False
use_local_model = False
verbose = False

use_cuda = not no_cuda and torch.cuda.is_available()
