from Dataset_Flores_Class import Dataset_Flores
from torchvision import transforms
import torchvision
import os

transform = transforms.Compose([
    torchvision.transforms.ToPILImage(mode=None),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = Dataset_Flores(root_path=os.path.join(os.getcwd(), "acumulados"), on_ram=False, test=False,
                          transform=transform)