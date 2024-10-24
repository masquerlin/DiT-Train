import torch
max_t = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epoch = 3000
batch_size = 6
model_save_path = './model/best.pth'

