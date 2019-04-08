import numpy as np
import argparse
import cv2
from PIL import Image
import chainer
from chainer import cuda, serializers, functions as F
from chainer_MyNet import MyNet
from transform_img import transform_estimation


chainer.using_config('enable_backprop', False)

class SideSmileDetector():
    def __init__(self, arch=None, weights_file=None, model=None, device=-1, n_out=2):
        print('Loading FaceNet...')
        self.model = arch(n_out)
        serializers.load_npz(weights_file, self.model)
        self.device = device
    def __call__(self, img):
        img_h, img_w, _ = img.shape

        # img = transform_estimation(img)
        resized_image = cv2.resize(img, (int(img_h/10), int(img_w/10)))
        x = np.array(resized_image[np.newaxis], dtype=np.float32).transpose(0, 3, 1, 2) / 256 - 0.5
        if self.device >= 0:
            x = cuda.to_gpu(x)

        y = self.model(x)
        if self.device >= 0:
            y = y.to_cpu()
        print(y.data)
        y = np.argmax(y.data)
        return y

def parse_args():
    parser = argparse.ArgumentParser(description='Train sidesmile estimation')
    parser.add_argument('--arch', default= MyNet, help='Model architecture')
    parser.add_argument('--weights', default='mynet.npz', help='weights file path')
    parser.add_argument('--img', help='image file path')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # load model
    sidesmile_detector = SideSmileDetector(args.arch, args.weights, device=args.gpu)

    # read image
    img = cv2.imread(args.img)
    # img = np.array(Image.open(args.img))
    # print(img)
    y = sidesmile_detector(img)

    if y:
        print("smile")
    else:
        print("not smile")
