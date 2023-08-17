import os

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

class NIH_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, n_TTA=0):
        self.data_dir = data_dir
        self.split = split
        self.n_TTA = n_TTA

        self.CLASSES = [
            'No Finding',  # shared
            'Infiltration',
            'Effusion',  # shared
            'Atelectasis',  # shared
            'Nodule',
            'Mass',
            'Consolidation',  # shared
            'Pneumothorax',  # shared
            'Pleural Thickening',
            'Cardiomegaly',  # shared
            'Emphysema',
            'Edema',  # shared
            'Fibrosis',
            'Subcutaneous Emphysema',  # shared
            'Pneumonia',  # shared
            'Tortuous Aorta',  # shared
            'Calcification of the Aorta',  # shared
            'Pneumoperitoneum',  # shared
            'Hernia',
            'Pneumomediastinum'  # shared
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'121722_nih-cxr-lt_labels_{split}.csv'))

        self.img_paths = self.label_df['id'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        self.labels = self.label_df[self.CLASSES].values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()

        if self.split == 'train':
            transform_list = [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        else:
            transform_list = [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])

        if self.split == 'test':
            x = cv2.resize(x, (256, 256))
        else:
            x = cv2.resize(x, (256, 256))

        if self.n_TTA > 0:
            x = torch.stack([self.tta_transform(x) for _ in range(self.n_TTA)], dim=0)
        else:
            x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).float()

class MIMIC_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, label_dir, split, n_TTA=0):
        self.split = split
        self.n_TTA = n_TTA

        self.CLASSES = [
            'Support Devices',
            'Lung Opacity',
            'Cardiomegaly',  # shared
            'Pleural Effusion',  # shared
            'Atelectasis',  # shared
            'Pneumonia',  # shared
            'Edema',  # shared
            'No Finding',  # shared
            'Enlarged Cardiomediastinum',
            'Consolidation',  # shared
            'Pneumothorax',  # shared
            'Fracture',
            'Calcification of the Aorta',  # shared
            'Tortuous Aorta',  # shared
            'Subcutaneous Emphysema',  # shared
            'Lung Lesion',
            'Pneumomediastinum',  # shared
            'Pneumoperitoneum',  # shared
            'Pleural Other'
        ]

        self.label_df = pd.read_csv(os.path.join(label_dir, f'121722_mimic-cxr-lt_labels_{split}.csv'))

        self.img_paths = self.label_df['path'].apply(lambda x: os.path.join(data_dir, x)).values.tolist()
        # self.img_paths = [f[:-4] + '_320.jpg' for f in self.img_paths]  # path to version of image pre-downsampled to 320x320
        self.labels = self.label_df[self.CLASSES].values

        self.cls_num_list = self.label_df[self.CLASSES].sum(0).values.tolist()


        if self.split == 'train':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])
        else:
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

        self.tta_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) )
            ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        x = cv2.imread(self.img_paths[idx])

        if self.split == 'test':
            x = cv2.resize(x, (256, 256))
        else:
            x = cv2.resize(x, (256, 256))

        if self.n_TTA > 0:
            x = torch.stack([self.tta_transform(x) for _ in range(self.n_TTA)], dim=0)
        else:
            x = self.transform(x)

        y = np.array(self.labels[idx])

        return x.float(), torch.from_numpy(y).float()