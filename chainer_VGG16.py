import chainer
from chainer import Chain
import chainer.links as L
import chainer.functions as F

class sideVGG16(chainer.Chain):
    def __init__(self, class_labels=2):
        super(sideVGG, self).__init__()

        with self.init_scope():
            self.base = L.VGG16Layers()
            self.fc6 = L.Linear(6144, 4096)
            self.fc7 = L.Linear(4096, 256)
            self.fc8 = L.Linear(256, class_labels)

    def __call__(self, x):
        h = self.base(x, layers=['pool5'])['pool5']
        h = F.dropout(F.relu(self.fc6(h)))
        h = F.dropout(F.relu(self.fc7(h)))
        return self.fc8(h)
