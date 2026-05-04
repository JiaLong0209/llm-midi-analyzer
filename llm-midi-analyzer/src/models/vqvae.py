import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualGRU(nn.Module):
    """GRU with residual connections and layer normalization."""
    def __init__(self, input_dim, hidden_dim, output_dim=None, num_layers=1, bidirectional=True):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, 
                          batch_first=True, bidirectional=bidirectional)
        
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.proj = nn.Linear(gru_out_dim, self.output_dim)
        self.ln = nn.LayerNorm(self.output_dim)
        self.shortcut = nn.Linear(input_dim, self.output_dim) if input_dim != self.output_dim else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out, _ = self.gru(x)
        out = self.proj(out)
        return self.ln(out + res)

class VectorQuantizer(nn.Module):
    """VQ layer with EMA updates, Random Restart, and LayerNorm."""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.5, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.normal_()
        
        # EMA parameters
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", torch.empty(num_embeddings, embedding_dim))
        self.ema_w.data.copy_(self.embedding.weight.data)
        
        self.decay = decay
        self.epsilon = epsilon
        self.ln = nn.LayerNorm(embedding_dim)

    def forward(self, inputs):
        # Apply LayerNorm to stabilize distribution
        inputs = self.ln(inputs)
        
        flat_input = inputs.reshape(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        
        # EMA updates
        if self.training:
            with torch.no_grad():
                self.ema_cluster_size.data.mul_(self.decay).add_(torch.sum(encodings, 0), alpha=1 - self.decay)
                
                # Laplace smoothing
                n = torch.sum(self.ema_cluster_size.data)
                self.ema_cluster_size.data.add_(self.epsilon).div_(n + self.num_embeddings * self.epsilon).mul_(n)
                
                dw = torch.matmul(encodings.t(), flat_input)
                self.ema_w.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)
                
                self.embedding.weight.data.copy_(self.ema_w / self.ema_cluster_size.unsqueeze(1))
                
                # Random Restart for Dead Codes
                usage = torch.sum(encodings, dim=0)
                dead_mask = usage == 0
                if dead_mask.any():
                    n_dead = dead_mask.sum().item()
                    if flat_input.size(0) >= n_dead:
                        indices = torch.randperm(flat_input.size(0), device=inputs.device)[:n_dead]
                        self.embedding.weight.data[dead_mask] = flat_input[indices]
                        self.ema_w.data[dead_mask] = flat_input[indices]
                        self.ema_cluster_size.data[dead_mask] = 1.0

        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices

class HierarchicalVQVAE(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=256, codebook_size=512, commitment_cost=1.0):
        super().__init__()
        # Encoder 1-Bar (projects to hidden_dim)
        self.enc_m1 = ResidualGRU(input_dim, hidden_dim, output_dim=hidden_dim, num_layers=2)
        self.vq_m1 = VectorQuantizer(codebook_size, hidden_dim, commitment_cost=commitment_cost)
        
        # Encoder 4-Bar (grouped by Conv1d)
        self.enc_m4_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=4, stride=4)
        self.enc_m4_ln = nn.LayerNorm(hidden_dim)
        self.vq_m4 = VectorQuantizer(codebook_size, hidden_dim, commitment_cost=commitment_cost)

        # Decoder (projects from hidden_dim back to input_dim)
        self.dec = ResidualGRU(hidden_dim, hidden_dim, output_dim=hidden_dim, num_layers=2)
        self.out_head = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Normalize inputs: 0-128 -> -1 to 1 approximately
        x_norm = x.float() / 128.0 - 1.0
        
        # m1 Stage
        m1_out = self.enc_m1(x_norm)
        q_m1, loss_m1, idx_m1 = self.vq_m1(m1_out)
        
        # m4 Stage
        m1_res = m1_out.transpose(1, 2)
        m4_out = F.elu(self.enc_m4_conv(m1_res)).transpose(1, 2)
        m4_out = self.enc_m4_ln(m4_out)
        q_m4, loss_m4, idx_m4 = self.vq_m4(m4_out)
        
        # Reconstruction Stage
        dec_out = self.dec(q_m1)
        recon_x = self.out_head(dec_out)
        
        # MSE on normalized values
        recon_loss = torch.mean((recon_x - x_norm)**2)
        
        # Denormalize for inspection
        recon_x_denorm = (recon_x + 1.0) * 128.0
        
        # Total loss (recon loss scaled for better gradient signal)
        total_loss = loss_m1 + loss_m4 + recon_loss * 10.0
        
        return recon_x_denorm, q_m1, q_m4, total_loss, idx_m1, idx_m4
