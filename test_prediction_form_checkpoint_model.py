import os
from Modelo import modelo
from Dataset_Flores_Class import Dataset_Flores
from torchvision import transforms
import torchvision
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from Transform import GetTransform


if __name__ == '__main__':
    transform = GetTransform()

    DS_Validation = Dataset_Flores(root_path=os.path.join(os.getcwd(), "acumulado"), on_ram=False, type='validation',
                                   transform=transform, shuffle=False, no_label=True)
    Validation_loader = DataLoader(DS_Validation, batch_size=64, num_workers=4)

    model = modelo.load_from_checkpoint(os.path.join('', 'log_antiguo_t2_sh1_test', 'checkpoints', 'epoch=4-step=95.ckpt'))
    model.freeze()

    print(DS_Validation[0].size())

    trainer = Trainer(accelerator='gpu')


    #blank_model = modelo()


    #model

    #model(DS_Validation[0])

    #print(model(DS_Validation[0]))


    #print(model.forward(DS_Validation[0]))


    predictions = trainer.predict(model, dataloaders=Validation_loader)
    #print('predicciones')
    #print(predictions)

    for _ in predictions:
        print(_)


