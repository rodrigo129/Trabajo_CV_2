import pandas as pd
from pathlib import Path
import os
from torch.utils.data import Dataset
import torch
import zipfile
import gdown
from PIL import Image


def fill_dataframe(root_path, folder, label, on_ram):
    """

    :param root_path: path of the dataset
    :param folder: folder of class to load
    :param label: label corresponding to the folder
    :param on_ram: flag that indicate if the image must be keep in ram
    :return: dataframe filled
    """
    folder_files = Path(os.path.join(root_path, folder))
    file_list = []
    file_label = []
    for file_path in sorted(folder_files.glob('*.jpg')):

        if on_ram:
            file_list.append(Image.open(os.fspath(file_path), "r"))
            # file_list.append(cv2.imread(os.fspath(file_path)))
        else:
            file_list.append(os.fspath(file_path))
        file_label.append(label)

    data = {'image': file_list, 'label': file_label}

    return pd.DataFrame(data)


def get_labels():
    """

    :return: dictionary of the class labels
    """
    return {0: 'rosas', 1: 'calas_rosa', 2: 'cardenales_rojas', 3: 'orejas_de_oso',
            4: 'otro'}


class FlowersDataset(Dataset):

    def __init__(self, root_path='', transform=None, dataset_type=False, on_ram=True, shuffle=True, no_label=False):

        """
            function that load the images of the dataset
            :param root_path: path of the folder containing the dataset
            :param transform: transform function for the images in the dataset
            :param dataset_type: dataset type to load ("train", "test" or "validation)
            :param on_ram: flag for keeping the images loaded in ram
            :param shuffle: flag for shuffle the images after load
            :param no_label: flag for hide the label of the image

            """

        self.on_ram = on_ram
        self.no_label = no_label
        self.transform = transform

        if not os.path.exists(root_path):
            print('Dataset Folder does not exist')

            if not os.path.exists('Dataset_Flores.zip') and not \
                    os.path.exists(os.path.join('Dataset_Flores', 'Dataset_Flores.zip')):
                # descargar
                print('Starting Download')
                url = 'https://drive.google.com/drive/u/1/folders/1Yj6LuftUGwk3rIEuYoA2UNrAyPtJ0v3b'
                gdown.download_folder(url, quiet=False, use_cookies=False)
                print('Download Finished ')

            if os.path.exists(os.path.join('Dataset_Flores', 'Dataset_Flores.zip')):
                os.rename(os.path.join('Dataset_Flores', 'Dataset_Flores.zip'), 'Dataset_Flores.zip')
                os.remove(os.path.join('Dataset_Flores',os.listdir('Dataset_Flores')[0]))
                os.rmdir('Dataset_Flores')

            if os.path.exists('Dataset_Flores.zip'):
                # descomprimir
                compressed_dataset = zipfile.ZipFile('Dataset_Flores.zip')
                print('Decompressing')
                compressed_dataset.extractall()
                compressed_dataset.close()

                os.remove('Dataset_Flores.zip')
                print('Done')
        else:
            print('Dataset Folder exist')

        # reemplazar por un diccionario
        if dataset_type == 'test':
            root_path = os.path.join(root_path, 'test')
        elif dataset_type == 'train':
            root_path = os.path.join(root_path, 'train')
        elif dataset_type == 'validation':
            root_path = os.path.join(root_path, 'validation')
        else:
            return

        df_Rosas = fill_dataframe(root_path, 'Rosas', 0, on_ram)
        df_Calas_rosa = fill_dataframe(root_path, 'Calas_rosa', 1, on_ram)
        df_Cardenales_rojas = fill_dataframe(root_path, 'Cardenales_rojas', 2, on_ram)
        df_Orejas_de_oso = fill_dataframe(root_path, 'Orejas_de_oso', 3, on_ram)
        df_Otros = fill_dataframe(root_path, 'Otros', 4, on_ram)

        self.Dataframe = pd.concat([df_Rosas, df_Calas_rosa, df_Cardenales_rojas, df_Orejas_de_oso, df_Otros]
                                   , ignore_index=True)

        if shuffle:
            self.Dataframe = self.Dataframe.sample(frac=1).reset_index(drop=True)

    def __len__(self):
        """
        function that give the length of the dataset

        :return: integer with the length of the dataset
        """
        return len(self.Dataframe.index)

    def __getitem__(self, idx):
        """
        function that return an image form the dataset

        :param idx: index of the image to obtain
        :return: image in the index
        """
        row = self.Dataframe.iloc[[idx]]
        label = torch.tensor(int(row.values[0][1]))

        if self.on_ram:
            img = row.values[0][0]
        else:
            img = Image.open(os.fspath(row.values[0][0]), "r")

        if self.transform:
            img = self.transform(img)

        if self.no_label:
            return img

        return img, label
