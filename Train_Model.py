
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.cuda import is_available


import os
from Transform import GetTransform
from Dataset_Flores_Class import Dataset_Flores
from Modelo import modelo
#import torch
print()


os.environ['TORCH_HOME'] = '.'
if __name__ == '__main__':
    # Definir Transformaciones
    transform = GetTransform()

    # Declarar Dataloades
    train_ds = Dataset_Flores(root_path=os.path.join(os.getcwd(), "acumulado"), on_ram=False, type='train',
                              transform=transform)
    test_ds = Dataset_Flores(root_path=os.path.join(os.getcwd(), "acumulado"), on_ram=False, type='test',
                             transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, num_workers=os.cpu_count())
    val_loader = DataLoader(test_ds, batch_size=64, num_workers=os.cpu_count())
    classes = ('Calas_rosa', 'Cardenales_rojas', 'Orejas_de_oso', 'Rosas',
               'Otors')
    print(f'train len {len(train_ds)}')
    print(f'test len {len(test_ds)}')

    # Declarar Modelo (Modelo en uso shufflenet_v2_x0_5)
    model = modelo()

    log_folder = os.path.join(os.getcwd(),'log')

    logger = TensorBoardLogger(log_folder)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(log_folder, 'checkpoints'),
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='val_loss',
                                          mode='min')

    # Declarar Entrenador
    trainer = Trainer(
        accelerator='gpu' if is_available() else 'cpu',
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # Iniciar Entrenamiento
    trainer.fit(model, train_loader, val_loader)
