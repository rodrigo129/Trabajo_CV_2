import pandas as pd
from pathlib import Path
import os
from torch.utils.data import Dataset
import cv2
import torch
import zipfile
import gdown
from PIL import Image


def cdf(root_path, folder, label, on_ram):
    folder_files = Path (os.path.join(root_path,folder))
    file_list = []
    file_label = []
    for file_path in sorted(folder_files.glob('*.jpg')):
        
        if on_ram:
            file_list.append(Image.open(os.fspath(file_path), "r"))
            #file_list.append(cv2.imread(os.fspath(file_path)))
        else:
            file_list.append(os.fspath(file_path))
        file_label.append(label)
    
    data = {'image':file_list, 'label':file_label}
    
    return pd.DataFrame(data)


class Dataset_Flores(Dataset):
    def __init__(self, root_path = '', transform = None, type=False, on_ram = True , shuffle = True ,no_label = False):

        """if not os.path.exists(root_path):
            print('np')

            """

        self.on_ram = on_ram
        self.no_label = no_label
        self.transform = transform


        if not os.path.exists(root_path):
            print('Dataset Folder does not exist')

            if not (os.path.exists('Dataset_Flores.zip') or os.path.exists(os.path.join('Dataset_Flores','Dataset_Flores.zip'))):
                #descargar
                print('Starting Download')
                url = 'https://drive.google.com/drive/u/1/folders/1Yj6LuftUGwk3rIEuYoA2UNrAyPtJ0v3b'
                gdown.download_folder(url, quiet=False, use_cookies=False)
                print('Download Finished ')

            if os.path.exists(os.path.join('Dataset_Flores', 'Dataset_Flores.zip')):
                os.rename(os.path.join('Dataset_Flores', 'Dataset_Flores.zip'), 'Dataset_Flores.zip')
                os.remove('Dataset_Flores')

            if os.path.exists('Dataset_Flores.zip'):
                #descomprimir
                compresed_dataset = zipfile.ZipFile('Dataset_Flores.zip')
                print('Decompressing')
                compresed_dataset.extractall()
                os.remove('Dataset_Flores.zip')
                print('Done')
        else:
            print('Dataset Folder exist')

        #reemplazar por un diccionario
        if type == 'test':
            root_path = os.path.join(root_path,'test')
        else:
            if type == 'train':
                root_path = os.path.join(root_path,'train')
            else:
                if type == 'validation':
                    root_path = os.path.join(root_path,'validation')
                else:
                    return

        df_Rosas = cdf(root_path, 'Rosas',0,on_ram)
        df_Calas_rosa = cdf(root_path, 'Calas_rosa',1,on_ram)
        df_Cardenales_rojas = cdf(root_path, 'Cardenales_rojas',2,on_ram)
        df_Orejas_de_oso = cdf(root_path, 'Orejas_de_oso',3,on_ram)
        df_Otros = cdf(root_path, 'Otros',4,on_ram)
        
        self.Dataframe = pd.concat([df_Rosas, df_Calas_rosa,df_Cardenales_rojas,df_Orejas_de_oso,df_Otros]
                                       , ignore_index=True)
        
        if shuffle:
            self.Dataframe = self.Dataframe.sample(frac=1).reset_index(drop=True)

        pass
    
    def __len__(self):
        return len(self.Dataframe.index)
    
    def __getitem__(self, idx):
        
        row = self.Dataframe.iloc[[idx]]
        
        numero = row.values[0][1]
        #print(numero)
        
        label = torch.tensor(int(numero)) 
        
        
        if self.on_ram:
            img = row.values[0][0]
        else:

            img = Image.open(os.fspath(row.values[0][0]), "r")

            #img = cv2.imread(row.values[0][0])
        #img = cv2.imread(img_path)
        
        if self.transform:
            img = self.transform(img)
        
        if self.no_label:
            return (img )


        return (img,label)