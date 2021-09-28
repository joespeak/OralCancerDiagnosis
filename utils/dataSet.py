import torch
from torch.utils.data import Dataset
import os
import config
import albumentations as A
from PIL import Image
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2



class MouthCancerSet(Dataset):
    def __init__(self, orginalPath, transforms=False):
        super(MouthCancerSet, self).__init__()
        self.normalImgs = []
        self.osscImgs = []

        normalPath = os.path.join(orginalPath, "normal")
        osscPath = os.path.join(orginalPath, "ossc")

        for img in os.listdir(normalPath):

            self.normalImgs.append((os.path.join(normalPath, img), 0))

        for img in os.listdir(osscPath):
            self.osscImgs.append((os.path.join(osscPath, img), 1))

        self.imgs = self.normalImgs + (self.osscImgs)
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.imgs[index][0]
        label = self.imgs[index][1]

        img = np.array(Image.open(img))

        if self.transforms:
            img = self.transforms(image=img)["image"]
            return img, label
        else:
            return ToTensorV2()(img), label

    def __len__(self):

        return len(self.imgs)

    def getNormalAndOssc(self):
        print(f"normal : {len(self.normalImgs)}")
        print(f"ossc : {len(self.osscImgs)}")

if __name__ == '__main__':
    ransforms = transforms = A.Compose([A.HorizontalFlip(p=0.5),
                                                A.VerticalFlip(p=0.5),
                                                A.RandomBrightness(limit=0.2, p=0.5),
                                                A.RandomContrast(limit=0.2, p=0.5),
                                                A.OneOf([
                                                    A.MotionBlur(blur_limit=5),
                                                    A.MedianBlur(blur_limit=5),
                                                    A.GaussianBlur(blur_limit=5),
                                                    A.GaussNoise(var_limit=(5.0, 30.0)),
                                                ], p=0.7),
                                                A.OneOf([
                                                    A.OpticalDistortion(distort_limit=1.0),
                                                    A.GridDistortion(num_steps=5, distort_limit=1.),
                                                    A.ElasticTransform(alpha=3),
                                                ], p=0.7),

                                                A.Resize(224, 224),
                                                # A.Cutout(max_h_size=int(224 * 0.375), max_w_size=int(224 * 0.375),
                                                #          num_holes=1, p=0.7),
                                                A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                                ToTensorV2()])
    mydata = MouthCancerSet(config.train_path, transforms)
    print(type(mydata[0][0]))
