import numpy as np

from PIL import Image
from pychu.utils import pair


class Compose:
    """Compose several transforms.

    Args:
        transforms(list): list of transforms
    """
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


###############################################################################
# PIL Image用の変換
###############################################################################


class Convert:
    def __init__(self, mode="RGB"):
        self.mode = mode

    def __call__(self, img):
        """色を変化させる

        Args:
            img (str): どの配色にするかを決める

        Returns:
            画像を返す
        """
        if self.mode != "RGB":
            color = self.mode
            img = img.convert("RGB")
            r, g, b = img.split()
            color_pallet = {"R": r, "G": g, "B": b}
            img = Image.merge("RGB", (color_pallet[color[0].upper()],
                                      color_pallet[color[1].upper()],
                                      color_pallet[color[2].upper()]))
            return img
        elif self.mode == "L":
            img = img.convert("L")
            return img
        else:
            return img.convert(self.mode)


class Resize:
    def __init__(self, size, mode=Image.BILINEAR):  # type: ignore
        self.size = pair(size)
        self.mode = mode
        self.mode = mode

    def __call__(self, img):
        return img.resize(self.size, self.mode)


class CenterCrop:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL:
    def __call__(self, array):
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass


###############################################################################
# numpy用の変換
###############################################################################


class Normalize:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    """Flatten a NumPy array.
    """
    def __call__(self, array):
        return array.flatten()


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=int):
        self.dtype = dtype
