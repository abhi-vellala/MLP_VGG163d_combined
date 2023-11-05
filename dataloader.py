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
        label = np.asarray(self.df.loc[index,'psa'])


        return self.DataTransformer(image), self.DataTransformer(features), label
    

class DataBuilder:

    def __init__(self, data_path, splitratio) -> None:
        self.df = pd.read_csv(data_path)
        self.splitratio = splitratio

    def prepare(self):
        seed = torch.initial_seed()
        
        trainlen = np.floor(self.df.shape[0]*self.splitratio).astype(int)
        trainidx = np.random.choice(range(df.shape[0]),trainlen, replace=False)
        traindata = self.df[self.df.index.isin(trainidx)].reset_index(drop=True)
        valdata = self.df[~self.df.index.isin(trainidx)].reset_index(drop=True)

        datasets = {"train":[], "validate":[]}
        for i in range(traindata.shape[0]):
            datasets['train'].append(BuildData(traindata)[i])
            
        for i in range(valdata.shape[0]):
            datasets['validate'].append(BuildData(valdata)[i])
            
        return datasets

    






if __name__ == '__main__':

    df = pd.read_csv('data_processed.csv')

    databuilder = DataBuilder('data_processed.csv', 0.8)

    datasets = databuilder.prepare()

    print(len(datasets['train'][0]))