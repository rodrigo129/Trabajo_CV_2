from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torchvision
import os

from Dataset_Flores_Class import Dataset_Flores
from Modelo import modelo

os.environ['TORCH_HOME'] = '.'
if __name__ == '__main__':


    transform = transforms.Compose([
        torchvision.transforms.ToPILImage(mode=None),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = Dataset_Flores(root_path=os.path.join(os.getcwd(),"Dataset"),on_ram = False,test=False, transform=transform)
    val_ds = Dataset_Flores(root_path=os.path.join(os.getcwd(),"Dataset"),on_ram = False,test=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=4)
    classes = ('Calas_rosa', 'Cardenales_rojas', 'Orejas_de_oso', 'Rosas',
               'Otors')
    print(f'train len {len(train_ds)}')
    print(f'val len {len(val_ds)}')





    # Init our model
    model = modelo()
    logger = TensorBoardLogger('./log')
    checkpoint_callback = ModelCheckpoint(dirpath='./log/checkpoints',
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='val_loss',
                                          mode='min')

    # Initialize a trainer
    trainer = Trainer(
        accelerator='gpu',
        max_epochs=5,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)
