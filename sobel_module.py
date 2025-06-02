# modules/sobel_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['StructuredSobelLayer']

class StructuredSobelLayer(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        init = torch.tensor([-1., -2., -1., 1., 2.], dtype=torch.float32)
        init = init.repeat(in_channels, 1)
        self.params = nn.Parameter(init)  # shape: (C, 5)

    def build_kernels(self):
        C = self.in_channels
        device = self.params.device
        kernels_x = torch.zeros(C, 3, 3, device=device)
        kernels_y = torch.zeros(C, 3, 3, device=device)

        for c in range(C):
            p = self.params[c]
            # Sobel X: 垂直偵測
            kx = torch.zeros(3, 3, device=device)
            # 上下負向對稱
            kx[0, 0] = -p[0]  # 左上
            kx[0, 1] = -p[1]  # 上中
            kx[0, 2] = -p[2]  # 右上
            kx[1, 0] = 0.0    # 中左
            kx[1, 1] = 0.0    # 中中
            kx[1, 2] = 0.0    # 中右
            kx[2, 0] = p[2]   # 左下（右上負向對稱）
            kx[2, 1] = p[1]   # 下中（上中負向對稱）
            kx[2, 2] = p[0]   # 右下（左上負向對稱）
            kernels_x[c] = kx

            # Sobel Y: 水平偵測
            ky = torch.zeros(3, 3, device=device)
            ky[0, 0] = -p[3]  # 左上
            ky[0, 1] = 0.0    # 上中
            ky[0, 2] = p[3]   # 右上（左上負向對稱）
            ky[1, 0] = -p[4]  # 中左
            ky[1, 1] = 0.0    # 中中
            ky[1, 2] = p[4]   # 中右（中左負向對稱）
            ky[2, 0] = -p[3]  # 左下（左上負向對稱）
            ky[2, 1] = 0.0    # 下中
            ky[2, 2] = p[3]   # 右下（右上負向對稱）
            kernels_y[c] = ky

        kernels_x = kernels_x / (kernels_x.norm(dim=(1,2), keepdim=True) + 1e-8)
        kernels_y = kernels_y / (kernels_y.norm(dim=(1,2), keepdim=True) + 1e-8)

        return kernels_x.unsqueeze(1), kernels_y.unsqueeze(1)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.in_channels
        kernels_x, kernels_y = self.build_kernels()
        edge_x = F.conv2d(x, kernels_x, padding=1, groups=C)
        edge_y = F.conv2d(x, kernels_y, padding=1, groups=C)
        return torch.cat([edge_x, edge_y], dim=1)  # (B, 2*C, H, W)
