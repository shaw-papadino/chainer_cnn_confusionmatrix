from chainer.datasets import LabeledImageDataset
from chainer.datasets import TransformDataset
from transform_img import *
import numpy as np

def split_dataset(fnamesp, fnamesn, index_labelsp, index_labelsn):
    permp = np.random.permutation(len(fnamesp))
    permn = np.random.permutation(len(fnamesn))

    split_p = np.split(np.array(fnamesp)[permp],2)
    t_indexp = np.split(np.array(index_labelsp)[permp],2)

    split_n = np.split(np.array(fnamesn)[permn],[int(len(fnamesn)/2)])
    t_indexn = np.split(np.array(index_labelsn)[permn],[int(len(fnamesn)/2)])

    permt = np.random.permutation(len(split_p[0][:9500])+len(split_n[0][:15000]))
    permv = np.random.permutation(len(split_p[1])+len(split_n[1]))
    d1 = LabeledImageDataset(list(zip(list(np.r_[split_p[0][:9500], split_n[0][:15000]][permt]),list(np.r_[t_indexp[0][:9500], t_indexn[0][:15000]][permt]))))
    d2 = LabeledImageDataset(list(zip(list(np.r_[split_p[1], split_n[1]][permv]),list(np.r_[t_indexp[1], t_indexn[1]][permv]))))

    train = TransformDataset(d1, transform)
    valid = TransformDataset(d2, transform)

    return train, valid
