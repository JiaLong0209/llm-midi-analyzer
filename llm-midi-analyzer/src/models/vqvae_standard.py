import torch
import torch.nn as nn
from .vqvae import VectorQuantizer

class VanillaVQVAE(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, codebook_size=512):
        super().__init__()
        
        # Encoder: 這裡使用簡單的 GRU 提取時序特徵
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Vector Quantizer
        self.vq = VectorQuantizer(codebook_size, hidden_dim, commitment_cost=0.25)
        
        # Decoder: 試圖還原原始輸入 (在此架構中為還原 token embedding 或 ids)
        # 這裡為了對稱性，使用 GRU 加上 Linear 投影回 input_dim
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, input_dim)
        
        # 1. Encode
        z_e, _ = self.encoder(x.float())
        
        # 2. Quantize
        z_q, loss, indices = self.vq(z_e)
        
        # 3. Decode
        z_d, _ = self.decoder(z_q)
        recon_x = self.fc_out(z_d)
        
        return recon_x, loss, indices

    def encode(self, x):
        z_e, _ = self.encoder(x.float())
        z_q, _, indices = self.vq(z_e)
        return z_q, indices
