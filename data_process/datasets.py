from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch 
import os 
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
# from torchvision import transforms
plt.ion() #interactive mode
label_names=['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation'] 
class XrayDataSet(Dataset):
    def __init__(self,csv_file="data_files/current_labels.csv",label_names=[],root_dir="images",transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations and user meta data.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
        self.label_names=label_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        image_path=os.path.join(self.root_dir,self.data.loc[idx,'Image Index'])
        label=torch.tensor(self.data.loc[idx,self.label_names].values.astype(int))
        image = Image.open(image_path).convert('RGB')
        # image=io.imread(image_path)
        if self.transform:
            image=self.transform(image)
        return image, label



