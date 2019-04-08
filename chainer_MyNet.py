import chainer
from chainer import Chain
import chainer.links as L
import chainer.functions as F

class MyNet(chainer.Chain):
  def __init__(self, n_out):
    super(MyNet, self).__init__()

    with self.init_scope():
      self.conv1 = L.Convolution2D(None, 16, 3, 1, 1)
      self.conv2 = L.Convolution2D(16, 32, 3, 1, 1)
      self.conv3 = L.Convolution2D(32, 64, 3, 1, 1)
      self.fc4 = L.Linear(None, 128)
      self.fc5 = L.Linear(128, n_out)

  def forward(self, x):

    h = F.relu(self.conv1(x))
    h = F.relu(self.conv2(h))
    h = F.max_pooling_2d(h, 2, 2)
    h = F.relu(self.conv3(h))
    h = F.max_pooling_2d(h, 2, 2)
    h = F.dropout(h, ratio=0.75)
    h = F.relu(self.fc4(h))
    h = F.dropout(h, ratio=0.75)
    return self.fc5(h)
