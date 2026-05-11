import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class SharedFrameEncoder(nn.Module):
    """
    Frame-level encoder with
    - Shallow CNN
    - Patch-level positional embedding
    - Pleural region emphasis
    - LayerNorm for stability
    """
    def __init__(self, in_channels=1, embed_dim=128, img_size=224, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.h_p = img_size // patch_size
        self.w_p = img_size // patch_size

        # Convolutional feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),  # no in-place

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, embed_dim, 3, 2, 1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=False),
        )

        # Positional Embeddings
        self.row_embed = nn.Embedding(self.h_p, embed_dim)
        self.col_embed = nn.Embedding(self.w_p, embed_dim)

        # LayerNorm for stability
        self.norm = nn.LayerNorm(embed_dim)

        # Initialization
        nn.init.normal_(self.row_embed.weight, mean=0, std=0.02)
        nn.init.normal_(self.col_embed.weight, mean=0, std=0.02)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.encoder(x)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D)

        N, D = x.shape[1], x.shape[2]
        device = x.device
        coords = torch.arange(N, device=device)
        rows = coords // self.w_p
        cols = coords % self.w_p
        pos = self.row_embed(rows) + self.col_embed(cols)
        x = x + pos.unsqueeze(0)

        x = self.norm(x)  # LayerNorm for stability
        x = x.view(B, T, N, D)
        return x


class SSAModule(nn.Module):
    def __init__(self, in_dim=128, num_heads=4, num_clusters=64, drop_stripe_prob=0.2, max_T=8):
        super().__init__()
        self.compress = SemanticCompressor(in_dim, num_clusters)
        self.attn = nn.MultiheadAttention(in_dim, num_heads, batch_first=True, dropout=0.1)
        self.ffn = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim)
        )
        self.drop_stripe_prob = drop_stripe_prob
        self.time_embedding = nn.Parameter(torch.randn(max_T, in_dim))
        nn.init.normal_(self.time_embedding, mean=0, std=0.02)

    def forward(self, z_masked, z_aux):
        B, T, N, D = z_aux.shape

        if self.training and torch.rand(1) < self.drop_stripe_prob:
            z_aux = z_aux.clone()
            stripe_idx = torch.randint(0, N, (1,)).item()
            z_aux[:, :, stripe_idx, :] = 0

        # Add temporal embedding
        time_emb = self.time_embedding.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D)
        time_emb = time_emb.expand(B, T, N, D)
        z_aux = z_aux + time_emb

        z_aux_flat = z_aux.reshape(B * T, N, D)
        z_anchor = self.compress(z_aux_flat)
        z_anchor = z_anchor.reshape(B, T, -1, D).reshape(B * T, -1, D)

        q = z_masked.reshape(B * T, -1, D)
        out, _ = self.attn(q, z_anchor, z_anchor)
        out = self.ffn(out)
        out = out.reshape(B, T, z_masked.shape[2], D)
        return out


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        return self.head(x)



class TeacherModel(nn.Module):
    def __init__(self, student: StudentModel, pre_train=False):
        super().__init__()
        self.encoder = deepcopy(student.encoder)
        self.proj = deepcopy(student.proj)
        self.pre_train = pre_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, mask_idx=None):
        x_feat = self.encoder(x)
        x_c = x_feat[:, 8:16]
        x_c = self.proj(x_c)
        if self.pre_train and mask_idx is not None:
            x_c = x_c.gather(
                dim=2,
                index=mask_idx.unsqueeze(-1).expand(-1, 8, -1, x_c.size(-1))
            )
        return x_c


class EMAUpdater:
    def __init__(self, momentum=0.996):
        self.momentum = momentum

    def update(self, student: nn.Module, teacher: nn.Module):
        with torch.no_grad():
            s_state = student.state_dict()
            t_state = teacher.state_dict()
            for name in t_state:
                if name in s_state:
                    t_state[name].copy_(
                        self.momentum * t_state[name] + (1 - self.momentum) * s_state[name]
                    )
