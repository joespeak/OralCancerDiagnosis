import torch
import torch.nn as nn
import cv2
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, batch_img, p=1):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        for i in range(len(batch_img)):
            img = batch_img[i, :, :, :]

            if np.random.uniform() < p:

                h = img.shape[1]
                w = img.shape[2]

                mask = np.ones((h, w), np.float32)

                for n in range(self.n_holes):
                    y = np.random.randint(h)
                    x = np.random.randint(w)

                    y1 = np.clip(y - self.length // 2, 0, h)
                    y2 = np.clip(y + self.length // 2, 0, h)
                    x1 = np.clip(x - self.length // 2, 0, w)
                    x2 = np.clip(x + self.length // 2, 0, w)

                    mask[y1: y2, x1: x2] = 0.

                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                batch_img[i, :, :, :] = img * mask

        return batch_img

if __name__ == '__main__':
    a = torch.FloatTensor(2,3,224,224)
    for i in range(len(a)):
        print(a[i, :, :, :].shape)