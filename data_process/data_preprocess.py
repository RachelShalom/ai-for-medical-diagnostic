import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold

# since I am going to train a binary model the idea is to understand if any of
# the situations accures , for that labels are converted to be binary
def to_binary_labels(x):
  if x in "No Finding":
    return 0
  else:
    return 1

def save_binary_labels(read_path="data_files/Data_Entry_2017_v2020.csv",save_path=""):
  label_names = ['Cardiomegaly', 
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
  labels=pd.read_csv(read_path)
  labels["labels_list"]=labels["Finding Labels"].apply(lambda x:x.split("|"))   
  labels["binary_labels"]=labels["Finding Labels"].apply(to_binary_labels) 
  df = pd.DataFrame(0, index=np.arange(len(labels)), columns=label_names)
  joined=labels[["Image Index","Patient ID","labels_list"]].join(df)
  for i, item in enumerate(joined["labels_list"]):
    joined.loc[i,item]=1
  joined.to_csv(save_path,index=False)
  return joined


# I didn't downoad all the x ray files 
# so saving the labels of the files that Idownloaded and ignoring the rest 
def save_current_labels(df,read_path="images",save_path=""):
    ''' get a df with all labels that exist and return a df with labels only for current downloaded images''' 
    files=os.listdir(read_path)
    file_names=pd.DataFrame(files, columns=["file_name"])
    labels=file_names.merge(df,how="left",left_on="file_name",right_on="Image Index")
    labels.to_csv(save_path,index=False)
    return labels



# Let's split the files to train validation and test by patient in order to prevent
# data leak ( i.e same patient is in test and train )
def split_data_and_save(file_name="data_files/current_labels.csv",n=5,files_to_save=["data_files/train.csv","data_files/test.csv"]):
  '''get data file and save train and test  or validation csv files
    Args:
      file_name (string): path for data file
      n (int): number of splits
      files_to_save (list of str) path of files to save
  '''
  patient_folds=GroupKFold(n_splits=n)
  data=pd.read_csv(file_name)
  X=data.loc[:,data.columns!="binary_labels"]
  y=data["binary_labels"]
  groups=data["Patient ID"]
  for i,(train_idx, test_idx) in enumerate(patient_folds.split(X,y,groups)):
    if i>0:
      break
    else:
      train=data.loc[train_idx,:]
      test=data.loc[test_idx,:]
      train.to_csv(files_to_save[0],index=False)
      test.to_csv(files_to_save[1],index=False)

def check_for_leaks(df1,df2,patient_col):
  '''
      Return True if there any patients are in both df1 and df2.

    Args:
        df1 (dataframe): dataframe describing first dataset
        df2 (dataframe): dataframe describing second dataset
        patient_col (str): string name of column with patient IDs
    
    Returns:
        leakage (bool): True if there is leakage, otherwise False
    '''
  unique_uers1=set(df1[patient_col])
  unique_users2=set(df2[patient_col])
  intersection=list(unique_uers1&unique_users2)
  if len(intersection)>0:
    print(f"same user found in both groups. users are: {intersection}")
    return True
  else:
    return False  


if __name__=="__main__":  
  save_current_labels(save_binary_labels(),"images")
  #split to train and test
  split_data_and_save(file_name="data_files/current_labels.csv",n=5,files_to_save=["data_files/train.csv","data_files/test.csv"])
  # split to test and validation
  split_data_and_save(file_name="data_files/test.csv",n=2,files_to_save=["data_files/test.csv","data_files/val.csv"])






