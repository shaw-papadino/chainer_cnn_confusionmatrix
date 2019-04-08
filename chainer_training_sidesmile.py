import numpy as np
from datetime import datetime
import argparse

import chainer
from chainer import cuda, optimizers, iterators, serializers, training, datasets
from chainer.cuda import to_cpu
from chainer.training import StandardUpdater, extensions, triggers
import chainer.links as L

from CMEvaluator import CMEvaluator
from split_dataset import split_dataset
from load_dataset import load_dataset
from chainer_MyNet import MyNet
from chainer_VGG16 import sideVGG16
# import chainer.cuda


def parse_args():
    parser = argparse.ArgumentParser(description='Train sidesmile estimation')
    parser.add_argument('--batchsize', '-B', type=int, default=16,
                        help='Training minibatch size')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU')
    parser.add_argument('--max_epoch', '-e', type=int, default=50,
                        help='Number of epoch to train')
    parser.add_argument('--cross_validation', '-c', type=int, default=1,
                        help='Number of cross_validation to train')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    max_epoch = args.max_epoch
    batchsize = args.batchsize
    cross = args.cross_validation
    gpu_id = args.gpu
    seed = int(datetime.now().timestamp())
    n_classes = 2 #len(dnames)
    np.random.seed(seed)

    if gpu_id >= 0:
        chainer.global_config.autotune = True
        chainer.cuda.set_max_workspace_size(512*1024*1024)
    else:
        chainer.using_config("use_ideep", "auto")

    chainer.print_runtime_info()
    print('GPU availability:', chainer.cuda.available)
    print('cuDNN availablility:', chainer.cuda.cudnn_enabled)

    fnamesp, fnamesn, index_labelsp, index_labelsn = load_dataset("side_all")
    for i in range(cross):

        print(i)

        model = L.Classifier(MyNet(n_classes))
        #model.base.disable_update()
        if gpu_id >= 0:
          cuda.get_device(gpu_id).use()
          model.to_gpu(gpu_id)

        else:
            pass
          # model.to_intel64()

        optimizer = optimizers.Adam()
        optimizer.setup(model)

        """
        1. ２分割シャッフル
        2. データセットにする
        3. トランスフォームで画像前処理
        """

        train, valid = split_dataset(fnamesp, fnamesn, index_labelsp, index_labelsn)
        print(len(valid))
        valid, _ = chainer.datasets.split_dataset_random(valid, int(len(valid)*0.1))
        print(len(train))
        print(len(valid))
        # print(train[0])
        train_iter = iterators.SerialIterator(train, batchsize, repeat=True, shuffle=True)
        valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)

        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='result')

        epoch_interval = (1, 'epoch')
        trainer.extend(CMEvaluator(valid_iter, model, device=gpu_id))
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
        trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}.npz'), trigger=epoch_interval)
        trainer.extend(extensions.ProgressBar(update_interval=1))
        trainer.extend(extensions.PrintReport( entries=['epoch', 'main/loss', 'main/accuracy', 'elapsed_time', 'cmrecall', 'cmaccuracy' ]))

        trainer.run()

    model.to_cpu()
    serializers.save_npz("mynet.npz", model)
    print("save is correct.")
