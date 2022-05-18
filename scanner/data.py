# pytorch-lightning data module

import os
import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
import torchio as tio
from pytorch_lightning import LightningDataModule

from .utils import RandomCrop


class STOICData(LightningDataModule):
    def __init__(self, data_path, seed, split_ratio, batch_size=8, num_workers=8):
        super().__init__()
        self.data_path = data_path
        # set runtime properties
        self.seed = torch.Generator().manual_seed(seed)
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # load images
        image_dir = os.path.join(self.data_path, 'data/uni/')
        self.images = [file for file in os.scandir(image_dir)]
        self.sep = int(len(self.images) * self.split_ratio)
        # load targets
        target_file = os.path.join(self.data_path, 'metadata/reference.csv')
        self.targets = pd.read_csv(target_file).set_index('PatientID')

    def setup(self, stage = None):
        # sample object constructor
        def _get_subject(image):
            patient_id = int(image.name.split('.')[0])
            prob_covid, prob_severe = self.targets.loc[patient_id].to_list()
            def _torch_load(pt):
                return torch.load(pt), None
            return tio.Subject(
                image=tio.ScalarImage(image.path, reader=_torch_load),
                target=[prob_severe, prob_covid - prob_severe, 1 - prob_covid]
            )
        # train/val split
        if stage in (None, "fit"):
            subjects = [_get_subject(image) for image in self.images]
            lengths = [self.sep, len(subjects) - self.sep]
            train_subjects, val_subjects = random_split(subjects, lengths, generator=self.seed)
            # set training stage transforms
            train_transform = tio.Compose([
                RandomCrop(size=(256, 256, 256)),
                tio.RandomAffine(scales=0.1, degrees=10, p=0.5),
                tio.RandomGamma(0.1, p=0.5)
            ])
            val_transform = tio.CropOrPad(256)
            self.train_dataset = tio.SubjectsDataset(train_subjects, transform=train_transform)
            self.val_dataset = tio.SubjectsDataset(val_subjects, transform=val_transform)
        # test stage
        if stage in (None, "test"):
            self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            shuffle=False
        )