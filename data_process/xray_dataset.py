from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch 
import os 
import matplotlib.pyplot as plt
from skimage import io
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
        image=io.imread(image_path)
        if self.transform:
            image=self.transform(image)
        return image, label 


def get_weighted_loss(pos_weights,neg_weights,epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        loss=0.0
      
        pos_loss=np.sum(np.mean(-pos_weights*y_true*np.log(y_pred+epsilon),axis=0))
        neg_loss=np.sum(np.mean(-(neg_weights)*(1-y_true)*np.log(1-y_pred+epsilon),axis=0))
        loss=pos_loss+neg_loss
        return loss
    return weighted_loss    
# transforms_=transforms.Compose([transforms.CenterCrop(500),transforms.ToTensor()])
if __name__=="__main__":

    xray_data=XrayDataSet(csv_file="data_files/current_labels.csv",label_names=label_names,root_dir="images")
    print(xray_data.__getitem__(0))
    print(xray_data.__len__())
    xray_dataloader=DataLoader(xray_data,batch_size=1,shuffle=True,num_workers=0)
    for i,(data, labels) in enumerate(xray_dataloader):
        print(data.shape, labels)
        if i>3:
            break
    # test the loss function
    y_true =np.array(
        [[1, 1, 1],
         [1, 1, 0],
         [0, 1, 0],
         [1, 0, 1]]
    )
    w_p = np.array([0.25, 0.25, 0.5])
    w_n = np.array([0.75, 0.75, 0.5])
    y_pred_1 =0.7*np.ones(y_true.shape)
    y_pred_2 =0.3*np.ones(y_true.shape)
    L = get_weighted_loss(w_p, w_n)

    
