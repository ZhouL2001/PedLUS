import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # student_output, teacher_output: (B, D)
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach()
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        loss = -torch.mean(torch.sum(teacher_out * student_out, dim=-1))

        # update center
        with torch.no_grad():
            batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        return loss

def temporal_consistency_loss(z_p_c, z_f_c):
    return F.mse_loss(z_p_c, z_f_c)

def temporal_smooth_loss(vc_feat):
    diffs = vc_feat[:, 1:] - vc_feat[:, :-1]
    # loss = torch.norm(diffs, dim=-1).pow(2).mean()
    loss = torch.mean(diffs ** 2)
    return loss

def total_loss(p_p_c, p_f_c, teacher_proj, z_p_c, z_f_c, vc_feat, device, alpha=2.0, beta=1.0, gamma=1.0):
    dino_loss = DINOLoss(out_dim=128).to(device)
    loss_dino_p = dino_loss(p_p_c, teacher_proj)
    loss_dino_f = dino_loss(p_f_c, teacher_proj)
    loss_temp = temporal_consistency_loss(z_p_c, z_f_c)
    loss_smooth = temporal_smooth_loss(vc_feat)

    loss = alpha * (loss_dino_p + loss_dino_f) + beta * loss_temp
    # loss = alpha * (loss_dino_p + loss_dino_f) + beta * loss_temp + gamma * loss_smooth

    return loss, loss_dino_p, loss_dino_f, loss_temp, loss_smooth

# finetune loss
