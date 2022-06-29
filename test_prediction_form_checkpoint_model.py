import os
from Modelo import modelo
from Dataset_Flores_Class import Dataset_Flores
from torchvision import transforms
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

if __name__ == '__main__':
    transform = transforms.Compose([
        torchvision.transforms.ToPILImage(mode=None),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    DS_Validation = Dataset_Flores(root_path=os.path.join(os.getcwd(), "acumulado"), on_ram=False, type='validation',
                                   transform=transform, shuffle=False)
    Validation_loader = DataLoader(DS_Validation, batch_size=64, num_workers=4)

    model = modelo.load_from_checkpoint(os.path.join('', 'log', 'checkpoints', 'epoch=0-step=10.ckpt'))

    trainer = Trainer()

    predictions = trainer.predict(model, dataloaders=Validation_loader)
    print(predictions)
