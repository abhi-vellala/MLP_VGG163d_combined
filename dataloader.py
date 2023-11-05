import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import nibabel as nib
from PIL import Image
import pandas as pd
from pathlib import Path


class BuildData(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame):

        self.df = df
        self.DataTransformer = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        image_path = Path(self.df.loc[index, 'image_path'])
        image = nib.load(image_path).get_fdata()

        features = np.asarray([self.df.loc[index,'age'],self.df.loc[index,'psa']]).reshape(1,2)
        print(features.shape)
        label = np.asarray(self.df.loc[index,'psa'])


        return self.DataTransformer(image), self.DataTransformer(features), label
    






if __name__ == '__main__':

    df = pd.read_csv('data_processed.csv')

    for i in range(3):
        img, features, label = BuildData(df, True, [224, 224])[i]
        print(img.shape)
        print(features)
        print(label)