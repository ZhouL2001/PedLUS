# #################################################################
import torch
import torch.nn as nn
import einops


class TemporalAwareAttention(nn.Module):
    """
    Lightweight Temporal-Aware Attention with Learnable Token
    - 3D Depthwise Conv
    - Temporal Token
    - Cross Attention
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, 3, 1, 1, groups=embed_dim),
            nn.BatchNorm3d(embed_dim),
            nn.ReLU(inplace=False),
        )

        self.temporal_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.temporal_token, std=0.02)

        self.mha = nn.MultiheadAttention(embed_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, feat):
        B, T, N, D = feat.shape
        S = int(N ** 0.5)

        feat2d = feat.view(B, T, S, S, D).permute(0, 4, 1, 2, 3)
        conv_feat = self.conv3d(feat2d)
        conv_feat = conv_feat.permute(0, 2, 1, 3, 4).flatten(3).permute(0, 1, 3, 2)

        global_token = self.temporal_token.expand(B, -1, -1)
        temporal_input = torch.cat([global_token, feat.mean(dim=2)], dim=1)

        attn_out, _ = self.mha(temporal_input, temporal_input, temporal_input)
        global_out = attn_out[:, 0]
        attn_feat = attn_out[:, 1:].unsqueeze(2).expand(-1, -1, N, -1)
        attn_feat = self.norm1(feat + attn_feat)

        ffn_out = self.ffn(attn_feat)
        out = self.norm2(attn_feat + ffn_out)
        return out, global_out


class VideoClassifier(nn.Module):
    def __init__(self, encoder, embed_dim=128, num_classes=4, clip_len=24, SA=True, TA=True):
        super().__init__()
        self.sa = SA
        self.ta = TA
        self.encoder = encoder
        self.clip_len = clip_len
        self.num_classes = num_classes

        # 原型参数
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)

        # 注意力模块
        self.temporal_fuser = nn.GRU(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)
        self.directional_attention = SpatialDirectionalAttention(embed_dim)
        self.temporal_attention = TemporalAwareAttention(embed_dim)

        self.drop = nn.Dropout(p=0.3)


    def forward(self, x, return_feat=False):
        B, T, C, H, W = x.shape
        clip_num = (T + self.clip_len - 1) // self.clip_len
        pad_len = clip_num * self.clip_len - T
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, C, H, W, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x = x.view(B * clip_num, self.clip_len, C, H, W)
        feat = self.encoder(x)  # (B*clip_num, clip_len, N, D)

        # Spatial Attention
        # if self.sa:
        #     weights = self.directional_attention(feat)
        #     feat = feat * weights + feat

        # ####################################################
        weights = self.directional_attention(feat)
        feat = feat * weights + feat
        feat_sa = self.drop(feat)
        # feat_sa = feat
        # ####################################################
        
        # ####################################################
        feat, global_tokens = self.temporal_attention(feat_sa)
        feat = feat + feat_sa
        feat = self.drop(feat)
        # feat = feat_sa
        # ####################################################

        # Pool over patches and frames
        feat = feat.mean(dim=2)  # (B*clip_num, clip_len, D)
        feat = feat.mean(dim=1)  # (B*clip_num, D)
        
        # fused_feat = feat + global_tokens
        
        fused_feat = feat

        # Further fuse with GRU
        fused_feat = fused_feat.view(B, clip_num, -1)
        fused, _ = self.temporal_fuser(fused_feat)
        pooled = fused.mean(dim=1)  # (B, D)

        #. #####################################
        # logits = self.classifier(pooled)
        #. #####################################

        # # 计算距离到原型
        logits = -torch.cdist(pooled, self.prototypes)  # (B, num_classes)
        if return_feat:
            return logits, pooled

        return logits