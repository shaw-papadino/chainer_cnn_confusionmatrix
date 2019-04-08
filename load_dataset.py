import os
import glob
from itertools import chain

def load_dataset(img_dir):
    # 画像フォルダ
    IMG_DIR = img_dir

    # 正解と不正解のフォルダ
    dnames = glob.glob('{}/*'.format(IMG_DIR))
    fnamesp = []
    fnamesn= []
    for i, d in enumerate(dnames):
        if i == 1:
            fnamesp.append(glob.glob('{}/*'.format(d)))
        elif i == 0:
            fnamesn.append(glob.glob('{}/*'.format(d)))


    fnamesp = list(chain.from_iterable(fnamesp))
    fnamesn = list(chain.from_iterable(fnamesn))

    print(len(fnamesp))
    # fnames = list(np.delete(np.array(fnames), [14049]))
    # print(len(fnames))

    #フォルダ名からラベル付け
    # front_positive, front_negative
    labelsp = []
    for f in fnamesp:
        labelsp.append(os.path.basename(os.path.dirname(f)))

    labelsn = []
    for f in fnamesn:
        labelsn.append(os.path.basename(os.path.dirname(f)))


    d_lists = []

    for i, d in enumerate(dnames):
            d_lists.append(os.path.basename(d))

    index_labelsp = []
    for l in labelsp:
        index_labelsp.append(d_lists.index(l))
    index_labelsn = []
    for l in labelsn:
        index_labelsn.append(d_lists.index(l))

    return fnamesp, fnamesn, index_labelsp, index_labelsn
