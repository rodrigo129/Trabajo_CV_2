import pandas as pd
import numpy as np
from pathlib import Path
import os
from torch.utils.data import Dataset
import cv2
import torch

def cdf(root_path, folder, label, on_ram):
    folder_files = Path (os.path.join(root_path,folder))
    file_list = []
    file_label = []
    for file_path in sorted(folder_files.glob('*.jpg')):
        
        if on_ram:
            file_list.append(cv2.imread(os.fspath(file_path)))
        else:
            file_list.append(os.fspath(file_path))
        file_label.append(label)
    
    data = {'image':file_list, 'label':file_label}
    
    return pd.DataFrame(data)


class Dataset_Flores(Dataset):
    def __init__(self, root_path = '',transform = None,test=False, on_ram = True ,shuffle = True ):
        self.on_ram = on_ram
        self.transform = transform
        
        
        if test:
            root_path = os.path.join(root_path,'test')
        else:
            root_path = os.path.join(root_path,'train')
        
        
        
        df_Rosas = cdf(root_path, 'Rosas',1,on_ram)
        df_Calas_rosa = cdf(root_path, 'Calas_rosa',2,on_ram)
        df_Cardenales_rojas = cdf(root_path, 'Cardenales_rojas',3,on_ram)
        df_Orejas_de_oso = cdf(root_path, 'Orejas_de_oso',4,on_ram)
        df_Otros = cdf(root_path, 'Otros',5,on_ram)
        
        self.Dataframe = pd.concat([df_Rosas, df_Calas_rosa,df_Cardenales_rojas,df_Orejas_de_oso,df_Otros]
                                       , ignore_index=True)
        
        if shuffle:
            self.Dataframe = self.Dataframe.sample(frac=1).reset_index(drop=True)
        
        
        
        
        
        pass
    
    def __len__(self):
        return len(self.Dataframe.index)
    
    def __getitem__(self, idx):
        
        row = self.Dataframe.iloc[[0]]
        
        numero = row.values[0][1]
        #print(numero)
        
        label = torch.tensor(int(numero)) 
        
        
        if self.on_ram:
            img = row.values[0][0]
        else:
            img = cv2.imread(row.values[0][0])
        #img = cv2.imread(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        
        return (img,label)