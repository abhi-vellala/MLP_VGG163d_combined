import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import nibabel as nib
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class BuildData(torch.utils.data.Dataset):

    def __init__(self, df: pd.DataFrame, target_depth):
        self.df = df
        self.DataTransformer = transforms.Compose([transforms.ToTensor()])
        self.target_depth = target_depth
        self.label_encoder = LabelEncoder().fit(list(self.df['grade'].unique()))

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        image_path = Path(self.df.loc[index, 'image_path'])
        image = nib.load(image_path).get_fdata()
        curr_depth = image.shape[2]
        # adding padding(images with 0s) to unify the depth of all the images across the data
        if curr_depth < self.target_depth:
            image = self.image_preprocessor(image, curr_depth)
        # Expand the dimensions to make it match (1,depth, h, w)
        # image = image[np.newaxis, :,:,:]
        print(torch.unsqueeze(self.DataTransformer(image),0).shape)
        features = np.asarray([self.df.loc[index,'age'],self.df.loc[index,'psa']]).reshape(1,2)

        
        label = self.label_encoder.transform([self.df.loc[index,'grade']])[0]

        return torch.unsqueeze(self.DataTransformer(image),0).float(), self.DataTransformer(features), label
    
    def image_preprocessor(self, image, curr_depth):

        padding = self.target_depth - curr_depth
        image_padded = np.pad(image, ((0, 0), (0, 0), (0, padding)), mode='constant')

        return image_padded


    

class DataBuilder:

    def __init__(self, data_path, splitratio) -> None:
        self.df = pd.read_csv(data_path)
        self.splitratio = splitratio
        self.target_depth = max(self.df['nimages'])

    def prepare(self):
        seed = torch.initial_seed()
        trainlen = np.floor(self.df.shape[0]*self.splitratio).astype(int)
        trainidx = np.random.choice(range(self.df.shape[0]),trainlen, replace=False)
        traindata = self.df[self.df.index.isin(trainidx)].reset_index(drop=True)
        valdata = self.df[~self.df.index.isin(trainidx)].reset_index(drop=True)

        datasets = {"train":[], "validate":[]}
        for i in range(traindata.shape[0]):
            datasets['train'].append(BuildData(traindata, self.target_depth)[i])
            
        for i in range(valdata.shape[0]):
            datasets['validate'].append(BuildData(valdata, self.target_depth)[i])
            
        return datasets

    






if __name__ == '__main__':

    df = pd.read_csv('data_processed.csv')

    databuilder = DataBuilder('data_processed.csv', 0.8)

    datasets = databuilder.prepare()

    print(len(datasets['train'][0]))