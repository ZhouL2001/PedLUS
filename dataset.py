import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


# Pretrain
class LUSVideoDataset(Dataset):
    def __init__(self, video_dir, clip_len=24, image_size=(224, 224), mask_ratio=0.4, patch_size=16):
        super().__init__()
        self.video_dir = Path(video_dir)
        self.clip_len = clip_len
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches_per_frame = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.num_mask_per_frame = int(self.num_patches_per_frame * self.mask_ratio)

        self.clips = self._generate_clips()

    def _generate_clips(self):
        clips = []
        for video_path in self.video_dir.glob("*.avi"):
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for start in range(0, total_frames - self.clip_len + 1):
                clips.append((str(video_path), start))
        return clips

    def _read_clip(self, path, start):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(self.clip_len):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, self.image_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()
        frames = np.stack(frames, axis=0)  # (T, H, W)
        return torch.from_numpy(frames).unsqueeze(1)  # (T, 1, H, W)

    def _generate_mask_idx(self):
        mask_idx_list = []
        for _ in range(8):  # 8帧
            indices = sorted(random.sample(range(self.num_patches_per_frame), self.num_mask_per_frame))
            mask_idx_list.append(torch.tensor(indices, dtype=torch.long))
        mask_idx = torch.stack(mask_idx_list)
        return mask_idx  # (8, num_mask_per_frame)
    
    def __len__(self):
        return len(self.clips)


    def __getitem__(self, idx):
        path, start = self.clips[idx]
        clip = self._read_clip(path, start)
        teacher_input = clip.clone()
        student_input = clip.clone()

        mask_idx = self._generate_mask_idx()
        return student_input, teacher_input, mask_idx


# Classifier
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path

# class LUSVideoClassificationDataset(Dataset):
#     def __init__(self, video_dir, image_size=(224, 224)):
#         super().__init__()
#         self.video_paths = list(Path(video_dir).glob('*.avi'))
#         self.image_size = image_size
#         self.samples = []
#         for path in self.video_paths:
#             label_char = path.stem[-1]  # 文件名最后一个字符
#             if label_char not in ['0', '1', '2', '3']:
#                 continue  # 过滤无效文件
#             label = int(label_char)
#             self.samples.append((str(path), label))

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         path, label = self.samples[index]
#         frames = self._read_video(path)
#         return frames, label, path  # 加入 path 返回

#     def _read_video(self, path):
#         cap = cv2.VideoCapture(path)
#         frames = []
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame = cv2.resize(frame, self.image_size)
#             frame = frame.astype(np.float32) / 255.0
#             frames.append(frame)
#         cap.release()
#         frames = np.stack(frames, axis=0)  # (T, H, W)
#         return torch.from_numpy(frames).unsqueeze(1)  # (T, 1, H, W)

# ^^^^^^ 
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class LUSVideoClassificationDataset(Dataset):
    def __init__(self, video_dir, image_size=(224, 224), use_augmentation=False):
        super().__init__()
        self.video_paths = list(Path(video_dir).glob('*.avi'))
        self.image_size = image_size
        self.augment = use_augmentation
        self.samples = []
        for path in self.video_paths:
            label_char = path.stem[-1]
            if label_char not in ['0', '1', '2', '3']:
                continue
            label = int(label_char)
            self.samples.append((str(path), label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, label = self.samples[index]
        frames = self._read_video(path)
        if self.augment:
            frames = self._augment_video(frames)
        return frames, label, path

    def _read_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, self.image_size)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        cap.release()
        frames = np.stack(frames, axis=0)  # (T, H, W)
        return torch.from_numpy(frames).unsqueeze(1)  # (T, 1, H, W)

    def _augment_video(self, video):
        """
        video: (T, 1, H, W)
        Apply the same transform to all frames
        """
        T, C, H, W = video.shape
        video_np = video.numpy().squeeze(1)  # (T, H, W)

        # --- Apply augmentations consistently to all frames ---
        if random.random() < 0.25:
            video_np = self._random_gamma(video_np)
        elif random.random() < 0.5:
            video_np = self._add_noise(video_np)
        elif random.random() < 0.75:
            video_np = self._random_blur(video_np)
        elif random.random() < 1.0:
            video_np = self._random_affine(video_np)

        video_np = np.clip(video_np, 0, 1)
        return torch.from_numpy(video_np).unsqueeze(1).float()

    def _random_gamma(self, frames, gamma_range=(0.8, 1.2)):
        gamma = random.uniform(*gamma_range)
        return frames ** gamma

    def _add_noise(self, frames, std=0.02):
        noise = np.random.randn(*frames.shape) * std
        return frames + noise

    def _random_blur(self, frames):
        blurred = []
        for f in frames:
            f_blur = cv2.GaussianBlur(f, (3, 3), 0)
            blurred.append(f_blur)
        return np.stack(blurred, axis=0)

    def _random_affine(self, frames):
        # Small translation and scale
        tx = random.uniform(-0.05, 0.05) * self.image_size[0]
        ty = random.uniform(-0.05, 0.05) * self.image_size[1]
        scale = random.uniform(0.95, 1.05)

        M = np.array([[scale, 0, tx],
                       [0, scale, ty]], dtype=np.float32)

        transformed = []
        for f in frames:
            f_t = cv2.warpAffine(f, M, (self.image_size[1], self.image_size[0]),
                                 borderMode=cv2.BORDER_REFLECT)
            transformed.append(f_t)
        return np.stack(transformed, axis=0)

# >>>>>>>>>>>>>>>>>>>>>>>
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# import random

# class LUSVideoClassificationDataset(Dataset):
#     def __init__(self, video_dir, image_size=(224, 224), use_augmentation=False):
#         """
#         use_augmentation=True 时，会把所有增强版本一起作为独立样本加入
#         """
#         super().__init__()
#         self.image_size = image_size
#         self.use_augmentation = use_augmentation

#         # 定义可用增强模式
#         self.augment_modes = ['original']
#         if self.use_augmentation:
#             self.augment_modes += ['gamma', 'noise', 'blur', 'affine']

#         self.samples = []
#         for path in Path(video_dir).glob('*.avi'):
#             label_char = path.stem[-1]
#             if label_char not in ['0', '1', '2', '3']:
#                 continue
#             label = int(label_char)

#             for mode in self.augment_modes:
#                 self.samples.append({
#                     'path': str(path),
#                     'label': label,
#                     'augment': mode
#                 })

#         print(f"[Dataset] Total samples after augmentation: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         item = self.samples[index]
#         path = item['path']
#         label = item['label']
#         augment_mode = item['augment']

#         frames = self._read_video(path)

#         if augment_mode != 'original':
#             frames = self._apply_augmentation(frames, mode=augment_mode)

#         return frames, label, path

#     def _read_video(self, path):
#         cap = cv2.VideoCapture(path)
#         frames = []
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame = cv2.resize(frame, self.image_size)
#             frame = frame.astype(np.float32) / 255.0
#             frames.append(frame)
#         cap.release()
#         frames = np.stack(frames, axis=0)  # (T, H, W)
#         return torch.from_numpy(frames).unsqueeze(1)  # (T, 1, H, W)

#     def _apply_augmentation(self, video, mode):
#         T, C, H, W = video.shape
#         video_np = video.numpy().squeeze(1)  # (T, H, W)

#         if mode == 'gamma':
#             video_np = self._random_gamma(video_np)
#         elif mode == 'noise':
#             video_np = self._add_noise(video_np)
#         elif mode == 'blur':
#             video_np = self._random_blur(video_np)
#         elif mode == 'affine':
#             video_np = self._random_affine(video_np)
#         else:
#             pass

#         video_np = np.clip(video_np, 0, 1)
#         return torch.from_numpy(video_np).unsqueeze(1).float()

#     def _random_gamma(self, frames, gamma_range=(0.7, 1.4)):
#         gamma = random.uniform(*gamma_range)
#         return frames ** gamma

#     def _add_noise(self, frames, std=0.02):
#         noise = np.random.randn(*frames.shape) * std
#         return frames + noise

#     def _random_blur(self, frames):
#         blurred = []
#         for f in frames:
#             f_blur = cv2.GaussianBlur(f, (5, 5), 0)
#             blurred.append(f_blur)
#         return np.stack(blurred, axis=0)

#     def _random_affine(self, frames):
#         tx = random.uniform(-0.05, 0.05) * self.image_size[0]
#         ty = random.uniform(-0.05, 0.05) * self.image_size[1]
#         scale = random.uniform(0.95, 1.05)
#         M = np.array([[scale, 0, tx],
#                       [0, scale, ty]], dtype=np.float32)

#         transformed = []
#         for f in frames:
#             f_t = cv2.warpAffine(f, M, (self.image_size[1], self.image_size[0]),
#                                  borderMode=cv2.BORDER_REFLECT)
#             transformed.append(f_t)
#         return np.stack(transformed, axis=0)



# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from pathlib import Path
# import random

# class LUSVideoClassificationDataset(Dataset):
#     def __init__(self, video_dir, image_size=(224, 224), use_augmentation=False):
#         """
#         use_augmentation=True 时，会把所有增强版本都加入
#         """
#         super().__init__()
#         self.image_size = image_size
#         self.use_augmentation = use_augmentation

#         # 预定义所有可用增强模式
#         self.all_augment_modes = [
#             'original',       # 保留原始
#             'gamma',
#             'contrast',
#             'noise',
#             'blur',
#             'affine',
#             'cutout',
#             'temporal_reverse'
#         ]

#         # 根据开关确定要用哪些
#         if self.use_augmentation:
#             self.augment_modes = self.all_augment_modes
#         else:
#             self.augment_modes = ['original']

#         # 构建完整样本列表
#         self.samples = []
#         for path in Path(video_dir).glob('*.avi'):
#             label_char = path.stem[-1]
#             if label_char not in ['0', '1', '2', '3']:
#                 continue
#             label = int(label_char)

#             for mode in self.augment_modes:
#                 self.samples.append({
#                     'path': str(path),
#                     'label': label,
#                     'augment': mode
#                 })

#         print(f"[Dataset] Total samples after augmentation: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, index):
#         item = self.samples[index]
#         path = item['path']
#         label = item['label']
#         augment_mode = item['augment']

#         frames = self._read_video(path)

#         if augment_mode != 'original':
#             frames = self._apply_augmentation(frames, mode=augment_mode)

#         return frames, label, path

#     def _read_video(self, path):
#         cap = cv2.VideoCapture(path)
#         frames = []
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame = cv2.resize(frame, self.image_size)
#             frame = frame.astype(np.float32) / 255.0
#             frames.append(frame)
#         cap.release()
#         frames = np.stack(frames, axis=0)  # (T, H, W)
#         return torch.from_numpy(frames).unsqueeze(1)  # (T, 1, H, W)

#     def _apply_augmentation(self, video, mode):
#         T, C, H, W = video.shape
#         video_np = video.numpy().squeeze(1)  # (T, H, W)

#         if mode == 'gamma':
#             video_np = self._random_gamma(video_np)
#         elif mode == 'contrast':
#             video_np = self._random_contrast(video_np)
#         elif mode == 'noise':
#             video_np = self._add_noise(video_np)
#         elif mode == 'blur':
#             video_np = self._random_blur(video_np)
#         elif mode == 'affine':
#             video_np = self._random_affine(video_np)
#         elif mode == 'cutout':
#             video_np = self._random_cutout(video_np)
#         elif mode == 'temporal_reverse':
#             video_np = video_np[::-1]
#         else:
#             pass

#         video_np = np.clip(video_np, 0, 1)
#         return torch.from_numpy(video_np).unsqueeze(1).float()

#     def _random_gamma(self, frames, gamma_range=(0.5, 1.8)):
#         gamma = random.uniform(*gamma_range)
#         return frames ** gamma

#     def _random_contrast(self, frames, factor_range=(0.7, 1.3)):
#         factor = random.uniform(*factor_range)
#         mean = frames.mean()
#         return (frames - mean) * factor + mean

#     def _add_noise(self, frames, std=0.05):
#         noise = np.random.randn(*frames.shape) * std
#         return frames + noise

#     def _random_blur(self, frames):
#         blurred = []
#         for f in frames:
#             if random.random() < 0.7:
#                 k = random.choice([3, 5])
#                 f_blur = cv2.GaussianBlur(f, (k, k), 0)
#             else:
#                 f_blur = f
#             blurred.append(f_blur)
#         return np.stack(blurred, axis=0)

#     def _random_cutout(self, frames, max_size=0.2):
#         T, H, W = frames.shape
#         for i in range(T):
#             if random.random() < 0.5:
#                 ch = int(H * max_size * random.uniform(0.5, 1))
#                 cw = int(W * max_size * random.uniform(0.5, 1))
#                 cy = random.randint(0, H - ch)
#                 cx = random.randint(0, W - cw)
#                 frames[i, cy:cy+ch, cx:cx+cw] = 0.0
#         return frames

#     def _random_affine(self, frames):
#         tx = random.uniform(-0.1, 0.1) * self.image_size[0]
#         ty = random.uniform(-0.1, 0.1) * self.image_size[1]
#         scale = random.uniform(0.9, 1.1)
#         M = np.array([[scale, 0, tx],
#                       [0, scale, ty]], dtype=np.float32)

#         transformed = []
#         for f in frames:
#             f_t = cv2.warpAffine(f, M, (self.image_size[1], self.image_size[0]),
#                                  borderMode=cv2.BORDER_REFLECT)
#             transformed.append(f_t)
#         return np.stack(transformed, axis=0)
