from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision.transforms.v2 import PILToTensor
class panda_data(Dataset):
    def __init__(self, data_path, label=0):
        super().__init__()
        self.data_path = data_path
        self.label = label
        self.imgs = [f for f in os.listdir(data_path) if f.endswith(('png', 'jpg', 'jpeg'))]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img = self.imgs[index]
        img_path = os.path.join(self.data_path, img)
        img_use =  Image.open(img_path)
        img_use = PILToTensor()(img_use) / 255.0
        return img_use, self.label


        