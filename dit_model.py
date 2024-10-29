import torch.nn as nn
import torch
import math
from torch import Tensor
from config import max_t, device
class TimeEmbedding(nn.Module):
    def __init__(self, emb_size, freq_emb_size=256):
        super().__init__()
        self.freq_emb_size = freq_emb_size
        self.mlp = nn.Sequential(
            nn.Linear(in_features=freq_emb_size, out_features=emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )
    
    def embedding_process(self, t, freq_emb_size, max_freq = 10000):
        half = freq_emb_size // 2
        freq = torch.exp(
            -math.log(max_freq)* torch.arange(start=0, end=half, dtype=torch.float32)
            /

            half).to(t.device)
        args = t[:,None] * freq[None]
        emb_out = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if freq_emb_size % 2:
            emb_out = torch.cat([emb_out, torch.zeros_like(emb_out[:,:1])], dim=-1)
        return emb_out
    
    def forward(self, t):
        time_freq = self.embedding_process(t, self.freq_emb_size)
        result = self.mlp(time_freq)
        return result


class DiTBlock(nn.Module):
    def __init__(self, emb_size, nhead):
        super().__init__()
        self.nhead = nhead
        self.emb_size = emb_size
        self.alfa_emb_1 = nn.Linear(emb_size, emb_size)
        self.gama_emb_1 = nn.Linear(emb_size, emb_size)
        self.beita_emb_1 = nn.Linear(emb_size, emb_size)
        self.alfa_emb_2 = nn.Linear(emb_size, emb_size)
        self.gama_emb_2 = nn.Linear(emb_size, emb_size)
        self.beita_emb_2 = nn.Linear(emb_size, emb_size)

        self.ln_1 = nn.LayerNorm(emb_size)
        self.ln_2 = nn.LayerNorm(emb_size)

        self.q_emb = nn.Linear(emb_size, self.nhead*emb_size)
        self.k_emb = nn.Linear(emb_size, self.nhead*emb_size)
        self.v_emb = nn.Linear(emb_size, self.nhead*emb_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4, emb_size)
        )
        self.linear_back = nn.Linear(self.nhead*emb_size, emb_size)
    
    def forward(self, x, con):
        y = self.ln_1(x)
        self.alfa_value_1 = self.alfa_emb_1(con).unsqueeze(1)
        self.gama_value_1 = self.gama_emb_1(con).unsqueeze(1)
        self.beita_value_1 = self.beita_emb_1(con).unsqueeze(1)
        self.alfa_value_2 = self.alfa_emb_2(con).unsqueeze(1)
        self.gama_value_2 = self.gama_emb_2(con).unsqueeze(1)
        self.beita_value_2 = self.beita_emb_2(con).unsqueeze(1)
        y = (1+self.gama_value_1) * y + self.beita_value_1

        q:Tensor = self.q_emb(y)
        k = self.k_emb(y)
        v = self.v_emb(y)
        q = q.view(q.size(0), q.size(1),self.nhead, self.emb_size).permute(0,2,1,3)
        k = k.view(k.size(0), k.size(1),self.nhead, self.emb_size).permute(0,2,3,1)
        v = v.view(v.size(0), v.size(1),self.nhead, self.emb_size).permute(0,2,1,3)
        attn=torch.softmax(q@k/math.sqrt(q.size(2)),dim=-1)
        y=attn@v
        y = y.permute(0,2,1,3)
        y = y.reshape(y.size(0), y.size(1), self.nhead*self.emb_size)
        y = self.linear_back(y)
        y = self.alfa_value_1*y
        y = x+y
        
        z = self.ln_2(y)
        z = (1+self.gama_value_2)*z + self.beita_value_2
        z = self.feed_forward(z)
        z = self.alfa_value_2 * z
        z = y + z
        return z
class DIT(nn.Module):
    def __init__(self, img_size, patch_size, channel, dit_num, head, label_num, emb_size) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.channel = channel
        self.dit_num = dit_num
        self.head = head
        self.label_num = label_num
        self.emb_size = emb_size
        self.patch_count = self.img_size // self.patch_size

        self.conv = nn.Conv2d(in_channels=channel, out_channels=channel*self.patch_size**2, kernel_size=patch_size, stride=patch_size, padding=0)
        self.label_emb = nn.Embedding(num_embeddings=self.label_num, embedding_dim=emb_size)
        self.patch_emb = nn.Linear(in_features=self.patch_size**2*channel, out_features=emb_size)
        self.time_emb = nn.Sequential(
            TimeEmbedding(emb_size),
            nn.Linear(emb_size,emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, emb_size)
        )
        self.position_emb = nn.Parameter(torch.rand(1, self.patch_count**2, emb_size))
        self.ln = nn.LayerNorm(emb_size)
        #变回最长那条
        self.linear = nn.Linear(emb_size, self.patch_size**2*channel)
        self.model_list = nn.ModuleList()
        for _ in range(dit_num):
            self.model_list.append(DiTBlock(emb_size, head))
    def forward(self, x:torch.Tensor, y, t):
        y_emb = self.label_emb(y)
        t_emb = self.time_emb(t)
        con = y_emb + t_emb

        x = self.conv(x) # batch, newchannel(patch_size * patch_size * channel), patch_count, patch_count
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), self.patch_count**2, x.size(3))
        x = self.patch_emb(x) # batch, patch_count*patch_count, embedding_size
        x = x + self.position_emb
        for dit in self.model_list:
            x = dit(x, con)
        x = self.ln(x)
        x = self.linear(x) # batch, patch_count*patch_count, self.patch_size**2*channel
        
        x = x.view(x.size(0), self.patch_count, self.patch_count, self.patch_size, self.patch_size, self.channel)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(x.size(0), self.channel, self.patch_count*self.patch_size, self.patch_count*self.patch_size)
        return x

class diffusion_part:
    def __init__(self) -> None:
        self.beita = torch.linspace(0.0001, 0.02, max_t)
        self.alpha = 1 - self.beita
        self.alpha_cumprod = torch.cumprod(self.alpha, dim=-1)
        self.alpha_cumprod_back = torch.cat((torch.tensor([1.0]), self.alpha_cumprod[:-1]), dim=-1)
        self.varance = (1-self.alpha)*(1-self.alpha_cumprod_back)/(1-self.alpha_cumprod)
    
    def forward_noise(self, x, t):
        noise = torch.randn_like(x)
        batch_alpha_cumprod = self.alpha_cumprod[t].view(x.size(0), 1, 1, 1)
        x = torch.sqrt(batch_alpha_cumprod)*x + torch.sqrt(1 - batch_alpha_cumprod) * noise
        return x, noise
    
    def denoise(self, model, x, y):
        steps = [x.clone()]
        x = x.to(device)
        varance = self.varance.to(device)
        alpha_cumprod = self.alpha_cumprod.to(device)
        alpha = self.alpha.to(device)
        y = y.to(device)
        model.eval()
        with torch.no_grad():
            for time in range(max_t-1, -1, -1):
                t = torch.full((x.size(0),), time).to(device)
                noise = model(x, y, t)

                temp = 1/torch.sqrt(alpha[t].view(x.size(0), 1, 1, 1))*(
                    x - (1-alpha[t].view(x.size(0), 1, 1, 1))/torch.sqrt(1-alpha_cumprod[t].view(x.size(0), 1, 1, 1))*noise
                )
                if time != 0:
                    x = temp + torch.randn_like(x) * torch.sqrt(varance[t].view(x.size(0), 1, 1, 1))
                else:
                    x = temp
                x = torch.clamp(x, -1.0, 1.0).detach()
                steps.append(x)
        return steps
