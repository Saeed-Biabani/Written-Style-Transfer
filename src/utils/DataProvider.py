from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from os import path
import torch
import json
import os


class DataGenerator(Dataset):
    def __init__(
        self,
        root,
        transforms = None
    ):
        super(DataGenerator, self).__init__()
        self.root = root

        self.imageDir = Path(
            path.join(
                self.root,
                "Images"
            )
        )
        self.targetDir = Path(
            path.join(
                self.root,
                "Targets"
            )
        )
        self.labelJson = Path(
            path.join(
                self.root,
                "labels.json"
            )
        )

        assert (
            self.labelJson.exists() and self.imageDir.exists()
        ), "datset directory not found!"

        self._images = list(
            self.imageDir.glob("*.jpg")
        )[:25600]
        self._labels = self.__loadlabels__()

        assert len(self._images) > 0

        self.transforms = transforms


    def __len__(self):
        return len(self._images)

    def __loadimage__(self, fname):
        return io.read_image(
            str(fname),
            mode = io.ImageReadMode.GRAY
        )

    def __loadlabels__(self):
        with open(
            self.labelJson,
            'r'
        ) as f:
            return json.load(f)

    def __getitem__(self, indx):
        if torch.is_tensor(indx):
            idx = idx.tolist()
            
        imageFname = self._images[indx]        
        image = self.__loadimage__(imageFname).float()
        
        targetFname = os.path.join(self.targetDir, os.path.basename(imageFname))
        target = self.__loadimage__(targetFname).float()
        label = self._labels[imageFname.stem]
        
        if self.transforms:
            image, target = self.transforms(image, target)
        
        return image, target#, label, len(label)