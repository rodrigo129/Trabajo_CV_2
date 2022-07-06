from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda import is_available
from Transform import GetTransform
from Dataset_Flores_Class import FlowersDataset
from NetworkModel import NetworkModel
import os
import typer
from pathlib import Path

os.environ['TORCH_HOME'] = '.'
app = typer.Typer()


def train_model(use_gpu: bool = typer.Option(True, help="Use GPU accelerator if is available, otherwise use CPU"),
                number_of_epoch: int = typer.Option(5, help="number of epoch to train the model", min=1),
                number_of_workers: int = typer.Option(os.cpu_count(), help="number of workers for the dataloader, de default "
                                                                    "value depends of the number of threads of the "
                                                                    "CPU", min=1),
                dataset_path: Path = typer.Option(Path(os.path.join(os.getcwd(), "acumulado")), help="Path where is the "
                                                                                              "dataset folder, "
                                                                                              "if the folder does not "
                                                                                              "exist, it will be "
                                                                                              "downloaded",
                                           file_okay=False),
                log_path: Path = typer.Option(Path(os.path.join(os.getcwd(), "log")),
                                       help="Path to save logs of the training")):
    """
    function that give train a neural network model to recognize roses, red calla lily, primrose, cardinal flower and
    'other' and save the results

    :param use_gpu: flag that indicate if it will try to use gpu acceleration
    :param number_of_epoch: number of epoch that the training will last
    :param number_of_workers: number of workers for the dataloader
    :param dataset_path: path of the dataset
    :param log_path: path for loging the results of the training
    :return:
    """
    dataset_path = os.path.join(os.getcwd(), os.fspath(dataset_path))
    log_path = os.path.join(os.getcwd(), os.fspath(log_path))
    # Define Transform
    transform = GetTransform()

    # Define Data loaders
    train_ds = FlowersDataset(root_path=dataset_path, on_ram=False, dataset_type='train',
                              transform=transform)
    train_loader = DataLoader(train_ds, batch_size=64, num_workers=number_of_workers)
    test_ds = FlowersDataset(root_path=os.path.join(os.getcwd(), "acumulado"), on_ram=False, dataset_type='test',
                             transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=number_of_workers)
    val_ds = FlowersDataset(root_path=os.path.join(os.getcwd(), "acumulado"), on_ram=False, dataset_type="validation",
                            transform=transform)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=number_of_workers)

    classes = ('Calas_rosa', 'Cardenales_rojas', 'Orejas_de_oso', 'Rosas',
               'Otors')
    print(f'train len {len(train_ds)}')
    print(f'test len {len(test_ds)}')

    # Declarar Modelo (Modelo en uso shufflenet_v2_x0_5)
    model = NetworkModel()

    log_folder = log_path

    logger = TensorBoardLogger(log_folder)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(log_folder, 'checkpoints'),
                                          save_top_k=1,
                                          verbose=True,
                                          monitor='val_loss',
                                          mode='min')

    # Declarar Entrenador
    trainer = Trainer(
        accelerator='gpu' if (is_available() and use_gpu) else 'cpu',
        max_epochs=number_of_epoch,
        logger=logger,
        callbacks=[checkpoint_callback]
    )

    # Iniciar Entrenamiento
    trainer.fit(model, train_loader, test_loader)

    model.model.eval()
    print('Testing Model with test dataset\n')

    correct_predictions = 0
    total_predictions = 0
    for img in test_ds:
        prediction = model.model(img[0].unsqueeze(0))
        prediction = prediction.cpu().detach().numpy()
        if prediction[0].argmax() == int(img[1]):
            correct_predictions += 1
        total_predictions += 1
    print(f'test dataset:\ncorrect predictions : {correct_predictions}\n total predictions {total_predictions}\n '
          f'accuracy : {correct_predictions / total_predictions}')

    print('Testing Model with validation dataset\n')
    correct_predictions = 0
    total_predictions = 0
    for img in val_ds:
        prediction = model.model(img[0].unsqueeze(0))
        prediction = prediction.cpu().detach().numpy()
        if prediction[0].argmax() == int(img[1]):
            correct_predictions += 1
        total_predictions += 1
    print(f'validation dataset:\ncorrect predictions : {correct_predictions}\n total predictions {total_predictions}\n '
          f'accuracy : {correct_predictions / total_predictions}')


if __name__ == '__main__':
    typer.run(train_model)
