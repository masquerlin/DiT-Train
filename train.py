from dit_model import DIT, diffusion_part
from config import *
from dataset import panda_data
import os
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
# 创建 TensorBoard writer
writer = SummaryWriter('./dit_training',filename_suffix='masquerlin_dit')

panda_data_set = panda_data(data_path='./data/Pandas')
print(len(panda_data_set))
diffusion_instance = diffusion_part()
model = DIT(img_size=256, patch_size=4, channel=3, dit_num=3, head=6, label_num=1, emb_size=64).to(device)
if os.path.exists(model_save_path):
    print('增量')
    model.load_state_dict(torch.load(model_save_path))

# 记录模型结构到 TensorBoard
dummy_input = torch.randn(1, 3, 256, 256).to(device)
dummy_label = torch.tensor([0]).to(device)
dummy_t = torch.randint(0, max_t, (1,)).to(device)
writer.add_graph(model, (dummy_input, dummy_label, dummy_t))


loss_fn = nn.L1Loss()
optimize = torch.optim.Adam(model.parameters(), lr=1e-6)
dataloader = DataLoader(dataset=panda_data_set, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=10)
model.train()


def visualize_batch(writer, images, title, step):
    """将一批图像可视化到TensorBoard"""
    if isinstance(images, list):
        images = torch.stack(images).cpu()
    images = images.squeeze(1)
    grid = torchvision.utils.make_grid(images, normalize=True)
    writer.add_image(title, grid, step)

def run_inference(model):
    """执行推理并保存结果"""
    with torch.no_grad():
        x=torch.randn(size=(1, 3,256,256))
        y=torch.tensor([0])   # 
        steps = diffusion_instance.denoise(model, x, y)
        final = []
        for i in range(max_t,0,-max_t//16):
            final_img=steps[i]
            final.append(final_img)
        final.reverse()
    model.train()
    return final    
    

global_step = 0
for epoch_use in range(0, epoch):
    # 使用 tqdm 包裹 dataloader
    if epoch_use == 0:
        print(model)
    epoch_loss = 0
    for image, label in tqdm(dataloader, desc=f'Epoch {epoch_use+1}/{epoch}', unit='batch'):
        x = image * 2 - 1
        y = label
        t = torch.randint(0, max_t, (image.size(0),))
        # print(t)
        x, noise = diffusion_instance.forward_noise(x, t)
        noise_pre = model(x.to(device), y.to(device), t.to(device))
        loss = loss_fn(noise.to(device), noise_pre)
        optimize.zero_grad()
        loss.backward()
        optimize.step()
        # 记录每个batch的损失
        writer.add_scalar('Loss/batch', loss.item(), global_step)
        # 记录一些中间结果的直方图
        if global_step % 100 == 0:
            writer.add_histogram('noise_prediction', noise_pre.detach().cpu(), 
                               global_step)
            writer.add_histogram('target_noise', noise.detach().cpu(), 
                               global_step)
        epoch_loss += loss.item()
        global_step += 1
    # 记录每个epoch的平均损失
    avg_epoch_loss = epoch_loss / len(dataloader)
    writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch_use)
    print(f'Epoch: {epoch_use + 1}, Average Loss: {avg_epoch_loss:.6f}')
    gen_images = run_inference(model)
    visualize_batch(writer, gen_images, 'Generated Images', epoch_use)
    torch.save(model.state_dict(), model_save_path)
writer.close()
    

