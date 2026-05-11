# # -*- coding: utf-8 -*-
# import os
# import matplotlib
# matplotlib.use("Agg")  # 无显示环境更稳
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import (
#     roc_auc_score, roc_curve, auc, f1_score, confusion_matrix,
#     precision_score, recall_score, precision_recall_curve, average_precision_score
# )
# from sklearn.preprocessing import label_binarize
# from sklearn.manifold import TSNE
# from torch.utils.data import DataLoader
# from argparse import ArgumentParser
# from model import StudentModel
# from classifier import VideoClassifier
# from dataset import LUSVideoClassificationDataset

# # ================= 样式 =================
# def use_paper_style():
#     plt.rcParams.update({
#         "pdf.fonttype": 42,   # 嵌入 TrueType，方便期刊/AI 编辑
#         "ps.fonttype": 42,
#         "font.sans-serif": ["DejaVu Sans", "Arial Unicode MS", "SimHei"],
#         "axes.titlesize": 18,
#         "axes.labelsize": 16,
#         "xtick.labelsize": 14,
#         "ytick.labelsize": 14,
#         "legend.fontsize": 13,
#     })

# # ================= 工具函数 =================
# def _safe_div(a, b):
#     return a / b if b != 0 else float('nan')

# def _nanmean(x):
#     arr = np.array(x, dtype=float)
#     return float(np.nanmean(arr))

# def _minmax(x: torch.Tensor):
#     x = x - x.min()
#     return x / (x.max() + 1e-6)

# # ================= 可视化：保存原始帧 =================
# def save_raw_frames_for_video(video_tensor, out_dir):
#     """
#     保存视频的每一帧为 PNG：
#     video_tensor: (T, C, H, W) torch.Tensor
#     out_dir: 保存目录，例如 fig/raw_frames/<video_name>/
#     """
#     os.makedirs(out_dir, exist_ok=True)
#     T, C, H, W = video_tensor.shape
#     for t in range(T):
#         frame = video_tensor[t].detach().cpu()
#         if C == 1:
#             base = _minmax(frame[0]).numpy()
#             plt.figure(figsize=(W/100, H/100), dpi=100)
#             plt.axis('off')
#             plt.imshow(base, cmap='gray')
#         else:
#             base = _minmax(frame).permute(1, 2, 0).numpy()
#             plt.figure(figsize=(W/100, H/100), dpi=100)
#             plt.axis('off')
#             plt.imshow(base)
#         out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
#         plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
#         plt.close()

# # ================= 可视化：热力图 =================
# def save_attention_maps_for_video(video_tensor, attn_maps, out_dir,
#                                   upsample='bicubic', normalize='video',
#                                   colormap='jet_plus', gamma=5.0, smooth=True):
#     """
#     Save attention maps with improved Jet+ colormap.
#     - Looks like 'jet' but less yellow, more balanced contrast.
#     - gamma > 1 -> suppress mid-range (reduces yellow dominance)
#     """
#     import os
#     import numpy as np
#     from PIL import Image, ImageFilter
#     import torch
#     import torch.nn.functional as F
#     import matplotlib.cm as cm
#     from matplotlib.colors import LinearSegmentedColormap

#     os.makedirs(out_dir, exist_ok=True)

#     if isinstance(attn_maps, np.ndarray):
#         attn_maps = torch.from_numpy(attn_maps)
#     attn = attn_maps.detach().cpu().float()

#     # normalize shape
#     if attn.ndim == 4:
#         if attn.shape[1] == 1:
#             attn = attn[:, 0]
#         else:
#             attn = attn.squeeze()
#     T, S, _ = attn.shape
#     H, W = int(video_tensor.shape[2]), int(video_tensor.shape[3])

#     eps = 1e-9
#     if normalize == 'video':
#         vals = attn.flatten().numpy()
#         vmin = np.percentile(vals, 1)
#         vmax = np.percentile(vals, 99)
#     elif normalize == 'frame':
#         vmin = vmax = None
#     else:
#         vmin, vmax = 0.0, 1.0

#     # ---- Custom Jet+ colormap ----
#     if colormap == 'jet_plus':
#         colors = [
#             (0.00, '#00007F'),  # deep blue
#             (0.25, '#007FFF'),  # medium blue
#             (0.50, '#00FFFF'),  # cyan (cool middle, replaces yellow)
#             (0.75, '#FF7F00'),  # orange-red
#             (1.00, '#7F0000')   # deep red
#         ]
#         cmap = LinearSegmentedColormap.from_list('jet_plus', colors, N=256)
#     else:
#         cmap = cm.get_cmap(colormap)

#     for t in range(T):
#         a = attn[t:t+1].unsqueeze(0)
#         up = F.interpolate(a, size=(H, W), mode=upsample, align_corners=False)[0, 0]

#         if normalize == 'frame':
#             lo, hi = float(a.min()), float(a.max())
#             up = (up - lo) / (hi - lo + eps)
#         elif normalize == 'video':
#             up = (up - vmin) / (vmax - vmin + eps)

#         up = up.clamp(0, 1)
#         up = torch.pow(up, gamma)

#         color_map = cmap(up.numpy())[:, :, :3]
#         img = (color_map * 255).astype(np.uint8)
#         pil_img = Image.fromarray(img)

#         if smooth:
#             pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.0))

#         pil_img.save(os.path.join(out_dir, f"attn_{t:04d}.png"))




# # def save_heatmaps_for_video(video_tensor, attn_maps, out_dir, alpha=0.35, cmap='jet'):
# #     """
# #     video_tensor: (T, C, H, W) torch.Tensor
# #     attn_maps:    (T, S, S)    torch.Tensor 或 np.ndarray
# #     """
# #     os.makedirs(out_dir, exist_ok=True)
# #     if isinstance(attn_maps, np.ndarray):
# #         attn_maps = torch.from_numpy(attn_maps)

# #     T, C, H, W = video_tensor.shape
# #     for t in range(T):
# #         frame = video_tensor[t].detach().cpu()              # (C,H,W)
# #         heat  = attn_maps[t].detach().cpu().float()         # (S,S)

# #         # 插值到原分辨率
# #         heat_up = F.interpolate(heat.unsqueeze(0).unsqueeze(0), size=(H, W),
# #                                 mode='bilinear', align_corners=False)[0, 0]
# #         heat_up = _minmax(heat_up)

# #         if C == 1:
# #             base = _minmax(frame[0])
# #             base_np = base.numpy()
# #             imshow_kwargs = dict(cmap='gray')
# #         else:
# #             base = _minmax(frame)
# #             base_np = base.permute(1, 2, 0).numpy()
# #             imshow_kwargs = {}

# #         # 叠加保存
# #         plt.figure(figsize=(W/100, H/100), dpi=100)
# #         plt.axis('off')
# #         plt.imshow(base_np, **imshow_kwargs)
# #         plt.imshow(heat_up.numpy(), cmap=cmap, alpha=alpha)
# #         plt.tight_layout(pad=0)
# #         out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
# #         plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
# #         plt.close()

# # ================= 可视化：热力图（不与原图融合） =================
# # （保留你的注释与旧版本，以便随时切换）
# # def save_heatmaps_for_video(...):
# #     ...

# def extract_attn_maps(model: VideoClassifier, videos: torch.Tensor,
#                       use_se_gate: bool = True, agg: str = "mean"):
#     """
#     生成更可靠的空间热图：使用 SpatialDirectionalAttention 的「前归一化能量图」
#     - 不经过 sigmoid / LayerNorm
#     - energy = (vert + horiz + diag)  [可选再乘以 SE gate]
#     - 通道聚合：mean 或 l2
#     返回: (B, T, S_vis, S_vis)，通常 S_vis=14（由 directional_attention.adapt_pool 决定）
#     """
#     try:
#         model.eval()
#         with torch.no_grad():
#             B, T, C, H, W = videos.shape
#             clip_len = getattr(model, "clip_len", None)
#             if clip_len is None:
#                 return None

#             clip_num = (T + clip_len - 1) // clip_len
#             pad_len = clip_num * clip_len - T
#             if pad_len > 0:
#                 pad = torch.zeros(B, pad_len, C, H, W, device=videos.device, dtype=videos.dtype)
#                 videos_pad = torch.cat([videos, pad], dim=1)
#             else:
#                 videos_pad = videos

#             x = videos_pad.view(B * clip_num, clip_len, C, H, W)
#             # encoder 输出: (B*clip_num, clip_len, N, D)
#             feat = model.encoder(x)

#             # ---- 还原成 (B*clip_len, D, S, S) 做卷积方向注意 ----
#             Bc, Tc, N, D = feat.shape
#             S = int(N ** 0.5)
#             feat2d = feat.reshape(Bc * Tc, S, S, D).permute(0, 3, 1, 2)  # (B*Tc, D, S, S)

#             da = model.directional_attention
#             v_feat = da.vert_conv(feat2d)
#             h_feat = da.horiz_conv(feat2d)
#             d_feat = da.diag_conv(feat2d)

#             # 能量图（未归一化、未LN）
#             energy = v_feat + h_feat + d_feat                # (B*Tc, D, S, S)
#             energy = da.adapt_pool(energy)                   # (B*Tc, D, S_vis, S_vis)

#             if use_se_gate:
#                 gate = da.se(energy)                         # (B*Tc, D, S_vis, S_vis)
#                 energy = energy * gate

#             # 通道聚合成单通道热图
#             if agg == "l2":
#                 heat = torch.norm(energy, dim=1, keepdim=True)      # (B*Tc, 1, S_vis, S_vis)
#             else:  # "mean"
#                 heat = energy.mean(dim=1, keepdim=True)             # (B*Tc, 1, S_vis, S_vis)

#             # 拼回完整时间序列并裁 padding
#             S_vis = heat.shape[-1]
#             heat = heat.view(B, clip_num, clip_len, 1, S_vis, S_vis)   # (B, clip, Tclip, 1, S_vis, S_vis)
#             heat = heat.reshape(B, clip_num * clip_len, S_vis, S_vis)  # (B, T_pad, S_vis, S_vis)
#             heat = heat[:, :T]                                         # (B, T, S_vis, S_vis)
#             return heat
#     except Exception as e:
#         print(f"[!] extract_attn_maps failed: {e}")
#         return None

# def plot_confusion_matrix_paper(
#     cm: np.ndarray,
#     class_names,
#     normalize: str = 'true',   # 'true'（按行归一化）, 'pred'（按列归一化）, None（原始计数）
#     show_counts: bool = True,
#     show_colorbar: bool = True,
#     cmap: str = 'Blues',
#     save_path: str = "fig/confusion_matrix.pdf"
# ):
#     cm = np.asarray(cm, dtype=float)
#     if normalize == 'true':
#         denom = cm.sum(axis=1, keepdims=True); denom[denom == 0] = 1.0
#         data = (cm / denom) * 100.0
#         fmt = lambda v: f"{v:.1f}%"; vmin, vmax = 0, 100
#     elif normalize == 'pred':
#         denom = cm.sum(axis=0, keepdims=True); denom[denom == 0] = 1.0
#         data = (cm / denom) * 100.0
#         fmt = lambda v: f"{v:.1f}%"; vmin, vmax = 0, 100
#     else:
#         data = cm; fmt = lambda v: f"{int(v)}"
#         vmin, vmax = 0, np.max(cm) if np.max(cm) > 0 else 1

#     plt.figure(figsize=(6.6, 5.8))
#     ax = sns.heatmap(
#         data, annot=False, cmap=cmap, vmin=vmin, vmax=vmax,
#         cbar=show_colorbar, square=True, linewidths=0.6, linecolor='white'
#     )

#     # 轴标题保持原样（不改动）
#     # ax.set_xlabel('Predicted', fontsize=14)
#     # ax.set_ylabel('True', fontsize=14)

#     # 轴刻度标签：统一调到 16
#     ax.set_xticks(np.arange(len(class_names)) + 0.5)
#     ax.set_yticks(np.arange(len(class_names)) + 0.5)
#     ax.set_xticklabels(class_names, fontsize=18, rotation=0)
#     ax.set_yticklabels(class_names, fontsize=18, rotation=0)

#     # 颜色条刻度：调到 16（如果有的话）
#     if show_colorbar:
#         cbar = ax.collections[0].colorbar
#         if cbar is not None:
#             cbar.ax.tick_params(labelsize=18)

#     for spine in ['top', 'right']:
#         ax.spines[spine].set_visible(False)

#     # 单元格内文字：统一调到 16
#     n = cm.shape[0]
#     norm_for_color = (data - vmin) / (vmax - vmin + 1e-9)
#     for i in range(n):
#         for j in range(n):
#             val_display = fmt(data[i, j])
#             if show_counts and normalize is not None:
#                 val_display = f"{val_display}\n({int(cm[i, j])})"
#             use_white = norm_for_color[i, j] > 0.5
#             color = 'white' if use_white else 'black'
#             fontweight = 'bold' if i == j else 'normal'
#             ax.text(
#                 j + 0.5, i + 0.5, val_display,
#                 ha='center', va='center',
#                 color=color, fontsize=20, fontweight=fontweight, linespacing=1.2
#             )

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, format='pdf', bbox_inches='tight')
#     plt.close()
#     print(f"[✔] Saved pretty confusion matrix to {save_path}")



# def plot_f1_radar(f1_scores, num_classes, save_path="fig/per_class_f1_radar.pdf"):
#     labels = [f'Class {i}' for i in range(num_classes)]
#     angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
#     scores = f1_scores.tolist()
#     scores += scores[:1]; angles += angles[:1]
#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
#     ax.plot(angles, scores, 'o-', linewidth=2); ax.fill(angles, scores, alpha=0.25)
#     ax.set_ylim(0, 1); ax.set_xticks([])
#     for angle, label in zip(angles[:-1], labels):
#         ax.text(angle, 1.12, label, fontsize=14, ha='center', va='center')
#     for i in range(num_classes):
#         ax.text(angles[i], scores[i] + 0.08, f"{scores[i]:.2f}",
#                 ha='center', va='center', fontsize=12, color='red')
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path); plt.close()
#     print(f"[✔] Radar chart saved to {save_path}")

# def plot_tsne(features, labels, num_classes, save_path="fig/tsne_vis.pdf"):
#     features = np.concatenate(features, axis=0); labels = np.array(labels)
#     tsne = TSNE(n_components=2, perplexity=15, learning_rate=100, n_iter=2000, init='pca', random_state=42)
#     tsne_result = tsne.fit_transform(features)

#     # ========= 正方形主图 + 紧凑底部图例 =========
#     from matplotlib.gridspec import GridSpec
#     fig = plt.figure(figsize=(8, 9))                 # 总体高度比之前略小
#     gs = GridSpec(nrows=2, ncols=1, height_ratios=[1.0, 0.18], hspace=0.01)

#     # ---- 顶部：正方形 t-SNE 主图 ----
#     ax = fig.add_subplot(gs[0])
#     palette = sns.color_palette("tab10", num_classes)
#     handles = []
#     for i in range(num_classes):
#         idx = labels == i
#         sc = ax.scatter(
#             tsne_result[idx, 0], tsne_result[idx, 1],
#             s=80, alpha=0.9, label=f"Class {i}",
#             edgecolors='black', linewidths=0.5, color=palette[i]
#         )
#         handles.append(sc)

#     # 去坐标 & 保证等比例 + 正方形视域
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.set_aspect('equal', adjustable='box')

#     # 强制正方形范围
#     x_min, x_max = np.min(tsne_result[:, 0]), np.max(tsne_result[:, 0])
#     y_min, y_max = np.min(tsne_result[:, 1]), np.max(tsne_result[:, 1])
#     x_c, y_c = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
#     span = max(x_max - x_min, y_max - y_min)
#     pad = 0.05 * span
#     half = span / 2.0 + pad
#     ax.set_xlim(x_c - half, x_c + half)
#     ax.set_ylim(y_c - half, y_c + half)

#     # ---- 底部：同宽图例区 ----
#     ax_leg = fig.add_subplot(gs[1])
#     ax_leg.axis('off')
#     ncol = min(num_classes, 6)    # 每行最多 6 列
#     ax_leg.legend(
#         handles=handles,
#         loc='center',
#         frameon=False,
#         ncol=ncol,
#         mode='expand',
#         handlelength=1.8,
#         handletextpad=0.6,
#         columnspacing=1.0,
#         labelspacing=0.5,
#         markerscale=1.6,
#         fontsize=16,
#         title=None
#     )

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()
#     print(f"[✔] t-SNE visualization saved to {save_path}")


# # ================= Inference =================
# def inference(model, dataloader, device, num_classes,
#               save_auc_fig=True, fig_path='fig/auc_curve.pdf', csv_path='result/predictions.csv',
#               save_heatmap=False, heatmap_alpha=0.35, heatmap_cmap='jet', heatmap_dir="fig/heatmaps",
#               save_raw_frames=False, raw_dir="fig/raw_frames"):
#     os.makedirs("fig", exist_ok=True)
#     os.makedirs(os.path.dirname(csv_path), exist_ok=True)
#     use_paper_style()

#     model.eval()
#     all_probs, all_labels, all_preds, all_feats, file_names = [], [], [], [], []
#     correct = 0
#     per_class_correct = np.zeros(num_classes, dtype=float)
#     per_class_total = np.zeros(num_classes, dtype=float)

#     with torch.no_grad():
#         for videos, labels, names in dataloader:
#             videos = videos.to(device)  # (B,T,C,H,W)
#             logits, pooled_feat = model(videos, return_feat=True)
#             probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
#             preds = logits.argmax(dim=1).cpu().numpy()
#             labels_np = labels.cpu().numpy()

#             all_probs.append(probs)
#             all_labels.append(labels_np)
#             all_preds.extend(preds)
#             all_feats.append(pooled_feat.cpu().numpy())
#             file_names.extend(names)

#             # # —— 保存原始帧 & 热力图（逐样本）——
#             # if save_raw_frames or save_heatmap:
#             #     attn_maps = None
#             #     if save_heatmap:
#             #         attn_maps = extract_attn_maps(model, videos)  # (B,T,S,S) or None

#             #     B, T, C, H, W = videos.shape
#             #     for b in range(B):
#             #         vid_name = str(names[b]) if isinstance(names, (list, tuple)) else str(names)
#             #         vid_stem = os.path.splitext(os.path.basename(vid_name))[0]

#             #         if save_raw_frames:
#             #             out_raw = os.path.join(raw_dir, vid_stem)
#             #             save_raw_frames_for_video(videos[b].detach().cpu(), out_raw)
#             #             print(f"[✔] Saved {T} raw frames to {out_raw}")

#             #         if save_heatmap:
#             #             if attn_maps is not None:
#             #                 out_hm = os.path.join(heatmap_dir, vid_stem)
#             #                 save_attention_maps_for_video(
#             #                     videos[b].detach().cpu(),          # (T,C,H,W)
#             #                     attn_maps[b].detach().cpu(),       # (T,S,S)
#             #                     out_hm, 
#             #                     # alpha=heatmap_alpha, 
#             #                     colormap=heatmap_cmap
#             #                 )
#             #                 print(f"[✔] Saved {attn_maps.shape[1]} heatmaps to {out_hm}")
#             #             else:
#             #                 print("[!] Could not extract attention maps (encoder/attention not accessible).")

#             # 统计
#             correct += (preds == labels_np).sum()
#             for i in range(len(labels_np)):
#                 per_class_total[labels_np[i]] += 1
#                 if preds[i] == labels_np[i]:
#                     per_class_correct[labels_np[i]] += 1

#     accuracy = _safe_div(correct, per_class_total.sum())
#     per_class_acc = np.array([_safe_div(per_class_correct[c], per_class_total[c]) for c in range(num_classes)])

#     probs = np.concatenate(all_probs, axis=0)
#     labels = np.concatenate(all_labels, axis=0)
#     labels_onehot = label_binarize(labels, classes=list(range(num_classes)))

#     # ===== ROC AUC =====
#     fpr, tpr, roc_auc = {}, {}, {}
#     for i in range(num_classes):
#         if labels_onehot[:, i].sum() > 0 and (1 - labels_onehot[:, i]).sum() > 0:
#             fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], probs[:, i])
#             roc_auc[i] = auc(fpr[i], tpr[i])
#         else:
#             fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), float('nan')
#     try:
#         micro_auc = roc_auc_score(labels_onehot, probs, average='micro')
#     except Exception:
#         micro_auc = float('nan')
#     try:
#         macro_auc = roc_auc_score(labels_onehot, probs, average='macro')
#     except Exception:
#         macro_auc = float('nan')

#     # ===== Precision/Recall/F1 =====
#     all_preds_concat = np.array(all_preds)
#     f1_per_class = f1_score(labels, all_preds_concat, average=None, zero_division=0)
#     macro_f1 = f1_score(labels, all_preds_concat, average='macro', zero_division=0)
#     micro_f1 = f1_score(labels, all_preds_concat, average='micro', zero_division=0)
#     macro_precision = precision_score(labels, all_preds_concat, average='macro', zero_division=0)
#     micro_precision = precision_score(labels, all_preds_concat, average='micro', zero_division=0)
#     macro_recall = recall_score(labels, all_preds_concat, average='macro', zero_division=0)
#     micro_recall = recall_score(labels, all_preds_concat, average='micro', zero_division=0)

#     # ===== 保存预测 CSV =====
#     pd.DataFrame({"SampleName": file_names, "TrueLabel": labels, "PredLabel": all_preds}).to_csv(csv_path, index=False)

#     # # ===== 混淆矩阵（论文风格）=====
#     cm = confusion_matrix(labels, all_preds_concat)
#     # class_names = [f"{i}" for i in range(num_classes)]
#     # plot_confusion_matrix_paper(cm, class_names, normalize='true', save_path="fig/confusion_matrix_rownorm.pdf")

#     # ===== Sensitivity / Specificity =====
#     tp = np.diag(cm).astype(float)
#     fn = cm.sum(axis=1) - tp
#     fp = cm.sum(axis=0) - tp
#     tn = cm.sum() - (tp + fn + fp)

#     sensitivity_per_class = np.array([_safe_div(tp[i], tp[i] + fn[i]) for i in range(num_classes)])
#     specificity_per_class = np.array([_safe_div(tn[i], tn[i] + fp[i]) for i in range(num_classes)])
#     balanced_acc_per_class = (sensitivity_per_class + specificity_per_class) / 2.0
#     macro_sensitivity = _nanmean(sensitivity_per_class)
#     macro_specificity = _nanmean(specificity_per_class)
#     macro_balanced_acc = _nanmean(balanced_acc_per_class)
#     overall_sensitivity = _safe_div(tp.sum(), tp.sum() + fn.sum())
#     overall_specificity = _safe_div(tn.sum(), tn.sum() + fp.sum())

#     # # ===== 保存敏感度/特异度 CSV =====
#     # metrics_rows = []
#     # for i in range(num_classes):
#     #     metrics_rows.append({
#     #         "Class": i,
#     #         "Sensitivity": sensitivity_per_class[i],
#     #         "Specificity": specificity_per_class[i],
#     #         "BalancedAccuracy": balanced_acc_per_class[i],
#     #         "TP": tp[i], "TN": tn[i], "FP": fp[i], "FN": fn[i]
#     #     })
#     # metrics_rows.append({
#     #     "Class": "Macro",
#     #     "Sensitivity": macro_sensitivity,
#     #     "Specificity": macro_specificity,
#     #     "BalancedAccuracy": macro_balanced_acc,
#     #     "TP": tp.sum(), "TN": tn.sum(), "FP": fp.sum(), "FN": fn.sum()
#     # })
#     # metrics_rows.append({
#     #     "Class": "Overall(OvR)",
#     #     "Sensitivity": overall_sensitivity,
#     #     "Specificity": overall_specificity,
#     #     "BalancedAccuracy": float('nan'),
#     #     "TP": tp.sum(), "TN": tn.sum(), "FP": fp.sum(), "FN": fn.sum()
#     # })
#     # os.makedirs("result", exist_ok=True)
#     # pd.DataFrame(metrics_rows).to_csv("result/metrics_summary.csv", index=False)

#     # ===== 其他可视化 =====
#     # plot_f1_radar(f1_per_class, num_classes)
#     # plot_tsne(all_feats, labels, num_classes)

#     # # ===== ROC（论文风格）=====
#     # if save_auc_fig:
#     #     plt.rcParams.update({
#     #         "legend.fontsize": 13,
#     #         "axes.titlesize": 18,
#     #         "axes.labelsize": 16,
#     #         "xtick.labelsize": 14,
#     #         "ytick.labelsize": 14,
#     #     })
#     #     plt.figure(figsize=(7.0, 5.4))
#     #     for i in range(num_classes):
#     #         if not np.isnan(roc_auc[i]):
#     #             plt.plot(fpr[i], tpr[i], lw=2.2, label=f'Class {i} (AUC={roc_auc[i]:.2f})')
#     #     plt.plot([0, 1], [0, 1], 'k--', lw=1.1)
#     #     plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
#     #     plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
#     #     ncol = 2 if num_classes >= 5 else 1
#     #     plt.legend(loc='lower right', frameon=False, ncol=ncol, handlelength=2.2, columnspacing=1.0, labelspacing=0.6)
#     #     plt.tight_layout()
#     #     os.makedirs(os.path.dirname(fig_path), exist_ok=True)
#     #     plt.savefig(fig_path, bbox_inches='tight'); plt.close()
#     #     print(f"[✔] ROC curve saved to {fig_path}")

#     # ===== 返回指标 =====
#     metrics = {
#         "accuracy": float(accuracy),
#         "per_class_accuracy": per_class_acc.tolist(),
#         "macro_auc": float(macro_auc),
#         "micro_auc": float(micro_auc),
#         "roc_auc_per_class": {int(k): (None if np.isnan(v) else float(v)) for k, v in roc_auc.items()},
#         "f1_per_class": f1_per_class.tolist(),
#         "macro_f1": float(macro_f1),
#         "micro_f1": float(micro_f1),
#         "macro_precision": float(macro_precision),
#         "micro_precision": float(micro_precision),
#         "macro_recall": float(macro_recall),
#         "micro_recall": float(micro_recall),
#         "sensitivity_per_class": sensitivity_per_class.tolist(),
#         "specificity_per_class": specificity_per_class.tolist(),
#         "balanced_accuracy_per_class": balanced_acc_per_class.tolist(),
#         "macro_sensitivity": float(macro_sensitivity),
#         "macro_specificity": float(macro_specificity),
#         "macro_balanced_accuracy": float(macro_balanced_acc),
#         "overall_sensitivity_OvR": float(overall_sensitivity),
#         "overall_specificity_OvR": float(overall_specificity),
#         "confusion_matrix": cm.tolist(),
#     }
#     return metrics

# # ================= Main =================
# def main():
#     parser = ArgumentParser()
#     parser.add_argument('--test_dir', type=str, default='data/test')
#     parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224))
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--device', type=str, default='cuda')
#     parser.add_argument('--clip_len', type=int, default=24)
#     parser.add_argument('--in_channels', type=int, default=1)
#     parser.add_argument('--embed_dim', type=int, default=128)
#     parser.add_argument('--num_classes', type=int, default=4)
#     parser.add_argument('--classifier_path', type=str, required=True)
#     parser.add_argument('--fig_save_path', type=str, default='fig/auc_curve.pdf')
#     parser.add_argument('--csv_save_path', type=str, default='result/predictions.csv')
#     # 原始帧/热力图开关与参数
#     parser.add_argument('--save_raw_frames', action='store_true', help='保存原始帧到单独目录')
#     parser.add_argument('--raw_dir', type=str, default='fig/raw_frames', help='原始帧输出根目录')
#     parser.add_argument('--save_heatmap', action='store_true', help='保存每帧注意力热力图')
#     parser.add_argument('--heatmap_alpha', type=float, default=0.35)
#     parser.add_argument('--heatmap_cmap', type=str, default='jet')
#     parser.add_argument('--heatmap_dir', type=str, default='fig/heatmaps')
#     args = parser.parse_args()

#     device = torch.device(args.device)
#     test_dataset = LUSVideoClassificationDataset(video_dir=args.test_dir, image_size=args.image_size)
#     test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#     student = StudentModel(embed_dim=args.embed_dim, use_temporal=True)
#     encoder = student.encoder
#     model = VideoClassifier(
#         encoder=encoder,
#         embed_dim=args.embed_dim,
#         num_classes=args.num_classes,
#         clip_len=args.clip_len).to(device)
#     model.load_state_dict(torch.load(args.classifier_path, map_location=device))

#     metrics = inference(
#         model, test_loader, device, num_classes=args.num_classes,
#         save_auc_fig=True, fig_path=args.fig_save_path, csv_path=args.csv_save_path,
#         save_heatmap=args.save_heatmap, heatmap_alpha=args.heatmap_alpha,
#         heatmap_cmap=args.heatmap_cmap, heatmap_dir=args.heatmap_dir,
#         save_raw_frames=args.save_raw_frames, raw_dir=args.raw_dir
#     )

#     print("\n=== Returned Metrics (keys) ===")
#     print(metrics)

# if __name__ == '__main__':
#     main()




# -*- coding: utf-8 -*-
import os
import time
import matplotlib
matplotlib.use("Agg")  # 无显示环境更稳
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, f1_score, confusion_matrix,
    precision_score, recall_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from model import StudentModel
from classifier import VideoClassifier
from dataset import LUSVideoClassificationDataset

# 可选：CPU 内存统计
try:
    import psutil
except ImportError:
    psutil = None

# 可选：MACs / FLOPs 统计
try:
    from thop import profile
except ImportError:
    profile = None


# ================= 样式 =================
def use_paper_style():
    plt.rcParams.update({
        "pdf.fonttype": 42,   # 嵌入 TrueType，方便期刊/AI 编辑
        "ps.fonttype": 42,
        "font.sans-serif": ["DejaVu Sans", "Arial Unicode MS", "SimHei"],
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    })


# ================= 工具函数 =================
def _safe_div(a, b):
    return a / b if b != 0 else float('nan')


def _nanmean(x):
    arr = np.array(x, dtype=float)
    return float(np.nanmean(arr))


def _minmax(x: torch.Tensor):
    x = x - x.min()
    return x / (x.max() + 1e-6)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_cost(model, input_shape, device):
    """
    input_shape: (B, T, C, H, W)
    返回:
        macs, flops
    说明:
        - 若安装了 thop，则计算 MACs
        - FLOPs 常近似为 2 * MACs
        - 若 thop 不可用，则返回 None
    """
    if profile is None:
        return None, None

    model.eval()
    dummy = torch.randn(*input_shape).to(device)

    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = 2 * macs if macs is not None else None
        return macs, flops
    except Exception as e:
        print(f"[!] Failed to compute MACs/FLOPs with thop: {e}")
        return None, None


def format_count(n):
    if n is None:
        return None
    n = float(n)
    if abs(n) >= 1e12:
        return f"{n / 1e12:.3f} T"
    elif abs(n) >= 1e9:
        return f"{n / 1e9:.3f} G"
    elif abs(n) >= 1e6:
        return f"{n / 1e6:.3f} M"
    elif abs(n) >= 1e3:
        return f"{n / 1e3:.3f} K"
    else:
        return f"{n:.0f}"


def get_cpu_memory_mb():
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 ** 2)


def reset_peak_memory_stats_if_possible(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def get_peak_memory_stats(device):
    """
    返回:
        peak_gpu_allocated_mb, peak_gpu_reserved_mb, cpu_mem_mb
    """
    cpu_mem_mb = get_cpu_memory_mb()

    if device.type == "cuda" and torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        return peak_allocated, peak_reserved, cpu_mem_mb
    else:
        return None, None, cpu_mem_mb


def synchronize_if_needed(device):
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


# ================= 可视化：保存原始帧 =================
def save_raw_frames_for_video(video_tensor, out_dir):
    """
    保存视频的每一帧为 PNG：
    video_tensor: (T, C, H, W) torch.Tensor
    out_dir: 保存目录，例如 fig/raw_frames/<video_name>/
    """
    os.makedirs(out_dir, exist_ok=True)
    T, C, H, W = video_tensor.shape
    for t in range(T):
        frame = video_tensor[t].detach().cpu()
        if C == 1:
            base = _minmax(frame[0]).numpy()
            plt.figure(figsize=(W/100, H/100), dpi=100)
            plt.axis('off')
            plt.imshow(base, cmap='gray')
        else:
            base = _minmax(frame).permute(1, 2, 0).numpy()
            plt.figure(figsize=(W/100, H/100), dpi=100)
            plt.axis('off')
            plt.imshow(base)
        out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
        plt.close()


# ================= 可视化：热力图 =================
def save_attention_maps_for_video(video_tensor, attn_maps, out_dir,
                                  upsample='bicubic', normalize='video',
                                  colormap='jet_plus', gamma=5.0, smooth=True):
    """
    Save attention maps with improved Jet+ colormap.
    - Looks like 'jet' but less yellow, more balanced contrast.
    - gamma > 1 -> suppress mid-range (reduces yellow dominance)
    """
    import os
    import numpy as np
    from PIL import Image, ImageFilter
    import torch
    import torch.nn.functional as F
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap

    os.makedirs(out_dir, exist_ok=True)

    if isinstance(attn_maps, np.ndarray):
        attn_maps = torch.from_numpy(attn_maps)
    attn = attn_maps.detach().cpu().float()

    # normalize shape
    if attn.ndim == 4:
        if attn.shape[1] == 1:
            attn = attn[:, 0]
        else:
            attn = attn.squeeze()
    T, S, _ = attn.shape
    H, W = int(video_tensor.shape[2]), int(video_tensor.shape[3])

    eps = 1e-9
    if normalize == 'video':
        vals = attn.flatten().numpy()
        vmin = np.percentile(vals, 1)
        vmax = np.percentile(vals, 99)
    elif normalize == 'frame':
        vmin = vmax = None
    else:
        vmin, vmax = 0.0, 1.0

    # ---- Custom Jet+ colormap ----
    if colormap == 'jet_plus':
        colors = [
            (0.00, '#00007F'),  # deep blue
            (0.25, '#007FFF'),  # medium blue
            (0.50, '#00FFFF'),  # cyan
            (0.75, '#FF7F00'),  # orange-red
            (1.00, '#7F0000')   # deep red
        ]
        cmap = LinearSegmentedColormap.from_list('jet_plus', colors, N=256)
    else:
        cmap = cm.get_cmap(colormap)

    for t in range(T):
        a = attn[t:t+1].unsqueeze(0)
        up = F.interpolate(a, size=(H, W), mode=upsample, align_corners=False)[0, 0]

        if normalize == 'frame':
            lo, hi = float(a.min()), float(a.max())
            up = (up - lo) / (hi - lo + eps)
        elif normalize == 'video':
            up = (up - vmin) / (vmax - vmin + eps)

        up = up.clamp(0, 1)
        up = torch.pow(up, gamma)

        color_map = cmap(up.numpy())[:, :, :3]
        img = (color_map * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)

        if smooth:
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=1.0))

        pil_img.save(os.path.join(out_dir, f"attn_{t:04d}.png"))


def extract_attn_maps(model: VideoClassifier, videos: torch.Tensor,
                      use_se_gate: bool = True, agg: str = "mean"):
    """
    生成更可靠的空间热图：使用 SpatialDirectionalAttention 的「前归一化能量图」
    - 不经过 sigmoid / LayerNorm
    - energy = (vert + horiz + diag)  [可选再乘以 SE gate]
    - 通道聚合：mean 或 l2
    返回: (B, T, S_vis, S_vis)，通常 S_vis=14（由 directional_attention.adapt_pool 决定）
    """
    try:
        model.eval()
        with torch.no_grad():
            B, T, C, H, W = videos.shape
            clip_len = getattr(model, "clip_len", None)
            if clip_len is None:
                return None

            clip_num = (T + clip_len - 1) // clip_len
            pad_len = clip_num * clip_len - T
            if pad_len > 0:
                pad = torch.zeros(B, pad_len, C, H, W, device=videos.device, dtype=videos.dtype)
                videos_pad = torch.cat([videos, pad], dim=1)
            else:
                videos_pad = videos

            x = videos_pad.view(B * clip_num, clip_len, C, H, W)
            feat = model.encoder(x)  # (B*clip_num, clip_len, N, D)

            Bc, Tc, N, D = feat.shape
            S = int(N ** 0.5)
            feat2d = feat.reshape(Bc * Tc, S, S, D).permute(0, 3, 1, 2)  # (B*Tc, D, S, S)

            da = model.directional_attention
            v_feat = da.vert_conv(feat2d)
            h_feat = da.horiz_conv(feat2d)
            d_feat = da.diag_conv(feat2d)

            energy = v_feat + h_feat + d_feat
            energy = da.adapt_pool(energy)

            if use_se_gate:
                gate = da.se(energy)
                energy = energy * gate

            if agg == "l2":
                heat = torch.norm(energy, dim=1, keepdim=True)
            else:
                heat = energy.mean(dim=1, keepdim=True)

            S_vis = heat.shape[-1]
            heat = heat.view(B, clip_num, clip_len, 1, S_vis, S_vis)
            heat = heat.reshape(B, clip_num * clip_len, S_vis, S_vis)
            heat = heat[:, :T]
            return heat
    except Exception as e:
        print(f"[!] extract_attn_maps failed: {e}")
        return None


def plot_confusion_matrix_paper(
    cm: np.ndarray,
    class_names,
    normalize: str = 'true',
    show_counts: bool = True,
    show_colorbar: bool = True,
    cmap: str = 'Blues',
    save_path: str = "fig/confusion_matrix.pdf"
):
    cm = np.asarray(cm, dtype=float)
    if normalize == 'true':
        denom = cm.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        data = (cm / denom) * 100.0
        fmt = lambda v: f"{v:.1f}%"
        vmin, vmax = 0, 100
    elif normalize == 'pred':
        denom = cm.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1.0
        data = (cm / denom) * 100.0
        fmt = lambda v: f"{v:.1f}%"
        vmin, vmax = 0, 100
    else:
        data = cm
        fmt = lambda v: f"{int(v)}"
        vmin, vmax = 0, np.max(cm) if np.max(cm) > 0 else 1

    plt.figure(figsize=(6.6, 5.8))
    ax = sns.heatmap(
        data, annot=False, cmap=cmap, vmin=vmin, vmax=vmax,
        cbar=show_colorbar, square=True, linewidths=0.6, linecolor='white'
    )

    ax.set_xticks(np.arange(len(class_names)) + 0.5)
    ax.set_yticks(np.arange(len(class_names)) + 0.5)
    ax.set_xticklabels(class_names, fontsize=18, rotation=0)
    ax.set_yticklabels(class_names, fontsize=18, rotation=0)

    if show_colorbar:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=18)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    n = cm.shape[0]
    norm_for_color = (data - vmin) / (vmax - vmin + 1e-9)
    for i in range(n):
        for j in range(n):
            val_display = fmt(data[i, j])
            if show_counts and normalize is not None:
                val_display = f"{val_display}\n({int(cm[i, j])})"
            use_white = norm_for_color[i, j] > 0.5
            color = 'white' if use_white else 'black'
            fontweight = 'bold' if i == j else 'normal'
            ax.text(
                j + 0.5, i + 0.5, val_display,
                ha='center', va='center',
                color=color, fontsize=20, fontweight=fontweight, linespacing=1.2
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"[✔] Saved pretty confusion matrix to {save_path}")


def plot_f1_radar(f1_scores, num_classes, save_path="fig/per_class_f1_radar.pdf"):
    labels = [f'Class {i}' for i in range(num_classes)]
    angles = np.linspace(0, 2 * np.pi, num_classes, endpoint=False).tolist()
    scores = f1_scores.tolist()
    scores += scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, scores, 'o-', linewidth=2)
    ax.fill(angles, scores, alpha=0.25)
    ax.set_ylim(0, 1)
    ax.set_xticks([])

    for angle, label in zip(angles[:-1], labels):
        ax.text(angle, 1.12, label, fontsize=14, ha='center', va='center')

    for i in range(num_classes):
        ax.text(angles[i], scores[i] + 0.08, f"{scores[i]:.2f}",
                ha='center', va='center', fontsize=12, color='red')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[✔] Radar chart saved to {save_path}")


def plot_tsne(features, labels, num_classes, save_path="fig/tsne_vis.pdf"):
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    tsne = TSNE(n_components=2, perplexity=15, learning_rate=100, n_iter=2000, init='pca', random_state=42)
    tsne_result = tsne.fit_transform(features)

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(8, 9))
    gs = GridSpec(nrows=2, ncols=1, height_ratios=[1.0, 0.18], hspace=0.01)

    ax = fig.add_subplot(gs[0])
    palette = sns.color_palette("tab10", num_classes)
    handles = []
    for i in range(num_classes):
        idx = labels == i
        sc = ax.scatter(
            tsne_result[idx, 0], tsne_result[idx, 1],
            s=80, alpha=0.9, label=f"Class {i}",
            edgecolors='black', linewidths=0.5, color=palette[i]
        )
        handles.append(sc)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')

    x_min, x_max = np.min(tsne_result[:, 0]), np.max(tsne_result[:, 0])
    y_min, y_max = np.min(tsne_result[:, 1]), np.max(tsne_result[:, 1])
    x_c, y_c = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    span = max(x_max - x_min, y_max - y_min)
    pad = 0.05 * span
    half = span / 2.0 + pad
    ax.set_xlim(x_c - half, x_c + half)
    ax.set_ylim(y_c - half, y_c + half)

    ax_leg = fig.add_subplot(gs[1])
    ax_leg.axis('off')
    ncol = min(num_classes, 6)
    ax_leg.legend(
        handles=handles,
        loc='center',
        frameon=False,
        ncol=ncol,
        mode='expand',
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.0,
        labelspacing=0.5,
        markerscale=1.6,
        fontsize=16,
        title=None
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[✔] t-SNE visualization saved to {save_path}")


# ================= Inference =================
def inference(model, dataloader, device, num_classes,
              save_auc_fig=True, fig_path='fig/auc_curve.pdf', csv_path='result/predictions.csv',
              save_heatmap=False, heatmap_alpha=0.35, heatmap_cmap='jet', heatmap_dir="fig/heatmaps",
              save_raw_frames=False, raw_dir="fig/raw_frames",
              measure_cost=True, warmup_steps=10):
    os.makedirs("fig", exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    use_paper_style()

    model.eval()
    all_probs, all_labels, all_preds, all_feats, file_names = [], [], [], [], []
    correct = 0
    per_class_correct = np.zeros(num_classes, dtype=float)
    per_class_total = np.zeros(num_classes, dtype=float)

    # ===== Computational cost / inference time / memory =====
    num_params, num_trainable_params = count_parameters(model)
    macs, flops = None, None
    batch_times = []
    timed_samples = 0

    first_batch_shape = None
    try:
        first_batch = next(iter(dataloader))
        first_videos = first_batch[0]
        first_batch_shape = tuple(first_videos.shape)  # (B, T, C, H, W)
    except Exception as e:
        print(f"[!] Could not inspect first batch for MACs/FLOPs: {e}")

    if measure_cost and first_batch_shape is not None:
        macs, flops = estimate_model_cost(model, first_batch_shape, device)

    reset_peak_memory_stats_if_possible(device)

    step_idx = 0
    with torch.no_grad():
        for videos, labels, names in dataloader:
            videos = videos.to(device)  # (B,T,C,H,W)
            batch_size_curr = videos.shape[0]

            if measure_cost:
                synchronize_if_needed(device)
                start_time = time.perf_counter()

            logits, pooled_feat = model(videos, return_feat=True)

            if measure_cost:
                synchronize_if_needed(device)
                end_time = time.perf_counter()
                if step_idx >= warmup_steps:
                    batch_times.append((end_time - start_time) * 1000.0)  # ms
                    timed_samples += batch_size_curr

            step_idx += 1

            probs = nn.functional.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_probs.append(probs)
            all_labels.append(labels_np)
            all_preds.extend(preds)
            all_feats.append(pooled_feat.cpu().numpy())
            file_names.extend(names)

            # # —— 保存原始帧 & 热力图（逐样本）——
            # if save_raw_frames or save_heatmap:
            #     attn_maps = None
            #     if save_heatmap:
            #         attn_maps = extract_attn_maps(model, videos)  # (B,T,S,S) or None
            #
            #     B, T, C, H, W = videos.shape
            #     for b in range(B):
            #         vid_name = str(names[b]) if isinstance(names, (list, tuple)) else str(names)
            #         vid_stem = os.path.splitext(os.path.basename(vid_name))[0]
            #
            #         if save_raw_frames:
            #             out_raw = os.path.join(raw_dir, vid_stem)
            #             save_raw_frames_for_video(videos[b].detach().cpu(), out_raw)
            #             print(f"[✔] Saved {T} raw frames to {out_raw}")
            #
            #         if save_heatmap:
            #             if attn_maps is not None:
            #                 out_hm = os.path.join(heatmap_dir, vid_stem)
            #                 save_attention_maps_for_video(
            #                     videos[b].detach().cpu(),
            #                     attn_maps[b].detach().cpu(),
            #                     out_hm,
            #                     colormap=heatmap_cmap
            #                 )
            #                 print(f"[✔] Saved {attn_maps.shape[1]} heatmaps to {out_hm}")
            #             else:
            #                 print("[!] Could not extract attention maps (encoder/attention not accessible).")

            correct += (preds == labels_np).sum()
            for i in range(len(labels_np)):
                per_class_total[labels_np[i]] += 1
                if preds[i] == labels_np[i]:
                    per_class_correct[labels_np[i]] += 1

    accuracy = _safe_div(correct, per_class_total.sum())
    per_class_acc = np.array([
        _safe_div(per_class_correct[c], per_class_total[c]) for c in range(num_classes)
    ])

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    labels_onehot = label_binarize(labels, classes=list(range(num_classes)))

    # ===== ROC AUC =====
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        if labels_onehot[:, i].sum() > 0 and (1 - labels_onehot[:, i]).sum() > 0:
            fpr[i], tpr[i], _ = roc_curve(labels_onehot[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), float('nan')

    try:
        micro_auc = roc_auc_score(labels_onehot, probs, average='micro')
    except Exception:
        micro_auc = float('nan')

    try:
        macro_auc = roc_auc_score(labels_onehot, probs, average='macro')
    except Exception:
        macro_auc = float('nan')

    # ===== Precision/Recall/F1 =====
    all_preds_concat = np.array(all_preds)
    f1_per_class = f1_score(labels, all_preds_concat, average=None, zero_division=0)
    macro_f1 = f1_score(labels, all_preds_concat, average='macro', zero_division=0)
    micro_f1 = f1_score(labels, all_preds_concat, average='micro', zero_division=0)
    macro_precision = precision_score(labels, all_preds_concat, average='macro', zero_division=0)
    micro_precision = precision_score(labels, all_preds_concat, average='micro', zero_division=0)
    macro_recall = recall_score(labels, all_preds_concat, average='macro', zero_division=0)
    micro_recall = recall_score(labels, all_preds_concat, average='micro', zero_division=0)

    # ===== 保存预测 CSV =====
    pd.DataFrame({
        "SampleName": file_names,
        "TrueLabel": labels,
        "PredLabel": all_preds
    }).to_csv(csv_path, index=False)

    # ===== 混淆矩阵 =====
    cm = confusion_matrix(labels, all_preds_concat)

    # ===== Sensitivity / Specificity =====
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (tp + fn + fp)

    sensitivity_per_class = np.array([
        _safe_div(tp[i], tp[i] + fn[i]) for i in range(num_classes)
    ])
    specificity_per_class = np.array([
        _safe_div(tn[i], tn[i] + fp[i]) for i in range(num_classes)
    ])
    balanced_acc_per_class = (sensitivity_per_class + specificity_per_class) / 2.0
    macro_sensitivity = _nanmean(sensitivity_per_class)
    macro_specificity = _nanmean(specificity_per_class)
    macro_balanced_acc = _nanmean(balanced_acc_per_class)
    overall_sensitivity = _safe_div(tp.sum(), tp.sum() + fn.sum())
    overall_specificity = _safe_div(tn.sum(), tn.sum() + fp.sum())

    # ===== 统计 memory/time =====
    peak_gpu_memory_mb, peak_gpu_reserved_mb, peak_cpu_memory_mb = get_peak_memory_stats(device)

    if len(batch_times) > 0:
        avg_inference_time_ms = float(np.mean(batch_times))
        std_inference_time_ms = float(np.std(batch_times))
        avg_inference_time_per_sample_ms = float(np.sum(batch_times) / max(timed_samples, 1))
    else:
        avg_inference_time_ms = None
        std_inference_time_ms = None
        avg_inference_time_per_sample_ms = None

    # ===== 返回指标 =====
    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc),
        # "micro_auc": float(micro_auc),
        # "roc_auc_per_class": {int(k): (None if np.isnan(v) else float(v)) for k, v in roc_auc.items()},
        # "f1_per_class": f1_per_class.tolist(),
        # "micro_f1": float(micro_f1),
        # "micro_precision": float(micro_precision),
        # "macro_recall": float(macro_recall),
        # "micro_recall": float(micro_recall),
        # "sensitivity_per_class": sensitivity_per_class.tolist(),
        # "specificity_per_class": specificity_per_class.tolist(),
        # "balanced_accuracy_per_class": balanced_acc_per_class.tolist(),
        "macro_sensitivity": float(macro_sensitivity),
        "macro_specificity": float(macro_specificity),
        "per_class_accuracy": per_class_acc.tolist(),
        # "macro_balanced_accuracy": float(macro_balanced_acc),
        # "overall_sensitivity_OvR": float(overall_sensitivity),
        # "overall_specificity_OvR": float(overall_specificity),
        # "confusion_matrix": cm.tolist(),

        # ===== Added: computational cost / inference time / memory =====
        "num_params": int(num_params),
        "num_trainable_params": int(num_trainable_params),
        "macs": None if macs is None else float(macs),
        "flops": None if flops is None else float(flops),
        "avg_inference_time_ms": avg_inference_time_ms,
        "std_inference_time_ms": std_inference_time_ms,
        "avg_inference_time_per_sample_ms": avg_inference_time_per_sample_ms,
        "peak_gpu_memory_mb": None if peak_gpu_memory_mb is None else float(peak_gpu_memory_mb),
        "peak_gpu_reserved_mb": None if peak_gpu_reserved_mb is None else float(peak_gpu_reserved_mb),
        "peak_cpu_memory_mb": None if peak_cpu_memory_mb is None else float(peak_cpu_memory_mb),
    }
    return metrics


# ================= Main =================
def main():
    parser = ArgumentParser()
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--image_size', type=int, nargs=2, default=(224, 224))
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip_len', type=int, default=24)
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--classifier_path', type=str, required=True)
    parser.add_argument('--fig_save_path', type=str, default='fig/auc_curve.pdf')
    parser.add_argument('--csv_save_path', type=str, default='result/predictions.csv')

    # 原始帧/热力图开关与参数
    parser.add_argument('--save_raw_frames', action='store_true', help='保存原始帧到单独目录')
    parser.add_argument('--raw_dir', type=str, default='fig/raw_frames', help='原始帧输出根目录')
    parser.add_argument('--save_heatmap', action='store_true', help='保存每帧注意力热力图')
    parser.add_argument('--heatmap_alpha', type=float, default=0.35)
    parser.add_argument('--heatmap_cmap', type=str, default='jet')
    parser.add_argument('--heatmap_dir', type=str, default='fig/heatmaps')
    parser.add_argument('--dataset_name', type=str, default='plus')

    # 复杂度 / 时间 / 内存
    parser.add_argument('--measure_cost', action='store_true', help='计算模型复杂度/推理时间/内存占用')
    parser.add_argument('--warmup_steps', type=int, default=10, help='计时前的预热 batch 数')

    args = parser.parse_args()

    device = torch.device(args.device)
    test_dataset = LUSVideoClassificationDataset(video_dir=args.test_dir, image_size=args.image_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    student = StudentModel(embed_dim=args.embed_dim, use_temporal=True)
    encoder = student.encoder
    model = VideoClassifier(
        encoder=encoder,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        clip_len=args.clip_len
    ).to(device)

    model.load_state_dict(torch.load(args.classifier_path, map_location=device))

    metrics = inference(
        model, test_loader, device, num_classes=args.num_classes,
        save_auc_fig=True, fig_path=args.fig_save_path, csv_path=args.csv_save_path,
        save_heatmap=args.save_heatmap, heatmap_alpha=args.heatmap_alpha,
        heatmap_cmap=args.heatmap_cmap, heatmap_dir=args.heatmap_dir,
        save_raw_frames=args.save_raw_frames, raw_dir=args.raw_dir,
        measure_cost=args.measure_cost, warmup_steps=args.warmup_steps
    )

    print("\n=== Evaluation Metrics ===")
    print(metrics)

    print("\n=== Model Complexity / Efficiency ===")
    print(f"Params: {metrics['num_params']} ({format_count(metrics['num_params'])})")
    print(f"Trainable Params: {metrics['num_trainable_params']} ({format_count(metrics['num_trainable_params'])})")
    print(f"MACs: {metrics['macs']} ({format_count(metrics['macs']) if metrics['macs'] is not None else 'None'})")
    print(f"FLOPs: {metrics['flops']} ({format_count(metrics['flops']) if metrics['flops'] is not None else 'None'})")
    print(f"Avg inference time / batch: {metrics['avg_inference_time_ms']} ms")
    print(f"Std inference time / batch: {metrics['std_inference_time_ms']} ms")
    print(f"Avg inference time / sample: {metrics['avg_inference_time_per_sample_ms']} ms")
    print(f"Peak GPU memory allocated: {metrics['peak_gpu_memory_mb']} MB")
    print(f"Peak GPU memory reserved: {metrics['peak_gpu_reserved_mb']} MB")
    print(f"Peak CPU memory: {metrics['peak_cpu_memory_mb']} MB")


if __name__ == '__main__':
    main()