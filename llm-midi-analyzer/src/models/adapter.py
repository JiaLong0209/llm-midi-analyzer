import torch
import torch.nn as nn

class MusicAlignmentAdapter(nn.Module):
    def __init__(self, vq_dim=256, llm_dim=2048):
        super().__init__()
        self.query_proj = nn.Linear(vq_dim, vq_dim) # From m4
        self.kv_proj = nn.Linear(vq_dim, vq_dim)    # From m1
        
        # Multi-head Cross-Attention
        self.attn = nn.MultiheadAttention(vq_dim, num_heads=8, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1)) # 門控機制穩定初始化
        self.final_proj = nn.Linear(vq_dim, llm_dim)

    def forward(self, m1_latent, m4_latent):
        # m4_latent 作為 Query，尋找其對應的 4 個 m1_latents 進行特徵融合
        # m1_latent: (Batch, 4, vq_dim)
        # m4_latent: (Batch, 1, vq_dim)
        q = self.query_proj(m4_latent)
        k = self.kv_proj(m1_latent)
        v = self.kv_proj(m1_latent)
        
        # 進行交叉注意力融合
        attn_out, _ = self.attn(query=q, key=k, value=v)
        
        # 殘差連線與門控機制後，投影至 LLM 維度 (如 Llama 3.2 1B 的 hidden_size)
        return self.final_proj(m4_latent + self.gate * attn_out)
