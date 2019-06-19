import os
import shutil
import socket
import stat
import subprocess
import torch
from zlib import crc32

import config


def str_to_float(s):
    return float(crc32(s.encode()) & 0xffffffff) / 2 ** 32


def matrix_to_tensor(matrix):
    tensor = torch.zeros(len(matrix), len(matrix[0]))
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            tensor[i][j] = value
    return tensor


def str_is_ip(ip):
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        # Not legal
        return False


def print_out(s, flag=0):
    if flag == 0:
        print('\033[0m' + s)
    else:
        print('\033[1;31;40m' + s)


def clear_tmp_dir():
    path = 'runs'
    if os.path.exists(path):
        for fileList in os.walk(path):
            for name in fileList[2]:
                os.chmod(os.path.join(fileList[0], name), stat.S_IWRITE)
                os.remove(os.path.join(fileList[0], name))
        shutil.rmtree(path)
        print('temp folder: "runs" is deleted')
    else:
        return 'no temp folder: "runs" found'


def kill_tensorboard():
    os.system('taskkill /IM tensorboard.exe /F')
    print('Tensorboard is killed')


def start_tensorboard():
    # DETACHED_PROCESS = 0x00000008
    CREATE_NO_WINDOW = 0x08000000
    dir = os.path.dirname(os.path.realpath(__file__))
    pid = subprocess.Popen(['tensorboard', '--logdir', '%s\\runs' % dir], creationflags=CREATE_NO_WINDOW).pid
    print('Tensorboard is started, pid = %d' % pid)


def get_data_path(file_id):
    return config.CACHE_DIR + file_id + '.csv'


def get_session_tag_path(file_id):
    return config.CACHE_DIR + file_id + '_session.tag.csv'


def get_session_guess_path(file_id):
    return config.CACHE_DIR + file_id + '_session.guess.csv'


def get_session_count_path(file_id):
    return config.CACHE_DIR + file_id + '_session.count.csv'


def get_metadata_path(file_id):
    return config.RAW_DIR + file_id + '.json'


def get_info_path(file_id):
    return config.CACHE_DIR + file_id + '.info.json'


def get_model_path(file_id):
    return config.CACHE_DIR + file_id + '.cnn.pt'


def get_loss_path(file_id):
    return config.CACHE_DIR + file_id + '.loss.pt'


if __name__ == '__main__':
    kill_tensorboard()
    start_tensorboard()
