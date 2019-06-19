import csv
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms

import config
import util

## Will be published later
train_file_id = ''
test_file_id = ''


def read_session_tags(path):
    res = []
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            res.append(int(row[1]))
    return res


def read_session_counts(path):
    res = []
    with open(path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            res.append(int(row[1]))
    return res


def get_file_map(file_id):
    file_map = []
    for name in os.listdir(config.IMAGE_DIR + file_id):
        base = os.path.splitext(name)[0]
        trunk_id, session_id = base.split('-')
        file_map.append([int(trunk_id), int(session_id)])
    file_map.sort()
    return file_map


class MyDataset(Dataset):
    def __init__(self, file_id):
        # xy = np.loadtxt('../dataSet/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        # self.x_data = torch.from_numpy(xy[:, 0:-1])
        # self.y_data = torch.from_numpy(xy[:, [-1]])
        # self.len = 1
        # self.trans = transforms.ToPILImage()
        self.file_id = file_id

        self.trans = transforms.ToTensor()
        self.tags = read_session_tags(util.get_session_tag_path(self.file_id))
        self.counts = read_session_counts(util.get_session_count_path(self.file_id))

        self.file_map = get_file_map(self.file_id)
        self.count = len(self.file_map)

    def __getitem__(self, index):
        trunk_id, session_id = self.file_map[index]
        img_path = config.IMAGE_DIR + '%s/%d-%d.png' % (self.file_id, trunk_id, session_id)
        img = Image.open(img_path)
        # img.show()
        tensor = self.trans(img)
        # print(type(img))
        # print(type(tensor))
        return tensor, self.tags[index], index

    def __len__(self):
        return self.count


if __name__ == '__main__':
    print(len(MyDataset(train_file_id)))
