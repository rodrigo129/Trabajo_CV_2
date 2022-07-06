from pytorch_lightning import LightningModule
import torchvision
import torch
import torchmetrics
from torch.nn import functional as F


class modelo(LightningModule):
    def __init__(self, show_model=False):
        super().__init__()
        self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=True, progress=True)
        self.model.fc = torch.nn.Sequential(torch.nn.Linear(1024, 5), torch.nn.Softmax(dim=1))
        if show_model:
            print(self.model)
        self.val_acc = torchmetrics.Accuracy(num_classes=5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self(x)
        loss = F.cross_entropy(out, y)
        print(out.size())
        self.val_acc(out, y.long())
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
