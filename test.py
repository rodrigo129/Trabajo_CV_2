from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision
import os

from Dataset_Flores_Class import Dataset_Flores
from Modelo import modelo

os.environ['TORCH_HOME'] = '.'
if __name__ == '__main__':

    #Definir Transformaciones
    transform = transforms.Compose([
        torchvision.transforms.ToPILImage(mode=None),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #Declarar Dataloades
    train_ds = Dataset_Flores(root_path=os.path.join(os.getcwd(),"acumulado"),on_ram = False,type='train', transform=transform)
    test_ds = Dataset_Flores(root_path=os.path.join(os.getcwd(),"acumulado"),on_ram = False,type='test', transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4)
    val_loader = DataLoader(test_ds, batch_size=32, num_workers=4)
    classes = ('Calas_rosa', 'Cardenales_rojas', 'Orejas_de_oso', 'Rosas',
               'Otors')
    print(f'train len {len(train_ds)}')
    print(f'test len {len(test_ds)}')

    #Declarar Modelo (Modelo en uso shufflenet_v2_x0_5)
    model = modelo()
    logger = TensorBoardLogger('./log')
    checkpoint_callback = ModelCheckpoint(dirpath='./log/checkpoints',
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='val_loss',
                                          mode='min')

    #Declarar Entrenador
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=5,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    #Iniciar Entrenamiento
    trainer.fit(model, train_loader, val_loader)
