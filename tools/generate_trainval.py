import os
import os.path as osp
from shutil import copyfile


def f_filter(f):
    if f[-4:] in ['.txt']:
        return True
    else:
        return False


input_dir = '/home/chip/datasets/FAIR1M/raw/validation/labelTxt/'
out_dir = '/home/chip/datasets/FAIR1M/raw/trainval/labelTxt/'

files = os.listdir(input_dir)
files = list(filter(f_filter, files))
files.sort(key=lambda files: int(files.split(".txt")[0]))

for i in range(len(files)):
    new_name = 16488 + int(files[i].split(".txt")[0])
    old_path = osp.join(input_dir, files[i])
    new_path = osp.join(out_dir, str(new_name) + ".txt")

    copyfile(old_path, new_path)
