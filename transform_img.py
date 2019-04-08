from PIL import Image
import numpy as np

def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):
    # マスクするかしないか
    if np.random.rand() > p:
        return image_origin
    image = np.copy(image_origin)

    # マスクする画素値をランダムで決める
    mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]

    # マスクのサイズとアスペクト比からマスクの高さと幅を決める
    # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image

def resize(img):
    # numpy to pil 配列の各値を1byte整数型(0～255)として画像のpixel値に変換
    # (channnel, width, height) -> (width, height, channel)

    img = Image.fromarray(img.transpose(1, 2, 0))
    width, height = img.size

    img = img.resize((int(width/10), int(height/10)), Image.BICUBIC)
    return np.asarray(img).transpose(2, 0, 1)

def transform(inputs):
    img, label = inputs
    # ... は後のやつ全部取るの意味
    # img = img[:3, ...]
    img = img[:3, ...]
    img = resize(img.astype(np.uint8))
    img = img.astype(np.float32)
    img = random_erasing(img)
    return img, label

def transform_estimation(inputs):
    img = inputs

    # ... は後のやつ全部取るの意味
    # img = img[:3, ...]
    img = img[:3]

    img = resize(img.astype(np.uint8))
    img = img.astype(np.float32)

    return img
