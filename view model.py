import torchvision
#print(torchvision.models.shufflenet_v2_x0_5(pretrained=False, progress=True))
#print(torchvision.models.alexnet(pretrained=True, progress=True))
import torch


print(torch.randn(20, 16, 50, 100).size())