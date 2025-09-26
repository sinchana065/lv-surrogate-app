# fno.py
# Fourier Neural Operator (2D) — channels-first (N, C, H, W)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _compl_mul2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex multiply using real/imag split weights.

    a: (B, m1, m2, Cin) complex
    b: (Cin, Cout, m1, m2, 2) real/imag
    returns: (B, m1, m2, Cout) complex
    """
    br = b[..., 0]   # (Cin, Cout, m1, m2)
    bi = b[..., 1]

    # permute to align with a
    br = br.permute(2, 3, 0, 1)  # (m1, m2, Cin, Cout)
    bi = bi.permute(2, 3, 0, 1)

    ar, ai = a.real, a.imag      # (B, m1, m2, Cin)

    real = torch.matmul(ar.unsqueeze(-2), br).squeeze(-2) - torch.matmul(ai.unsqueeze(-2), bi).squeeze(-2)
    imag = torch.matmul(ar.unsqueeze(-2), bi).squeeze(-2) + torch.matmul(ai.unsqueeze(-2), br).squeeze(-2)

    return torch.complex(real, imag)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / math.sqrt(in_channels)
        self.weight_pos = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, 2) * scale
        )
        self.weight_neg = nn.Parameter(
            torch.randn(in_channels, out_channels, modes1, modes2, 2) * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="forward")
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1,
                             dtype=x_ft.dtype, device=x.device)

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        a = x_ft[:, :, :m1, :m2]
        w = self.weight_pos[:, :, :m1, :m2, :]
        a_perm = a.permute(0, 2, 3, 1)  # (B, m1, m2, Cin)
        out_top = _compl_mul2d(a_perm, w)  # (B, m1, m2, Cout)
        out_ft[:, :, :m1, :m2] = out_top.permute(0, 3, 1, 2)

        a2 = x_ft[:, :, -m1:, :m2]
        w2 = self.weight_neg[:, :, :m1, :m2, :]
        a2_perm = a2.permute(0, 2, 3, 1)
        out_bot = _compl_mul2d(a2_perm, w2)
        out_ft[:, :, -m1:, :m2] = out_bot.permute(0, 3, 1, 2)

        x_out = torch.fft.irfft2(out_ft, s=(H, W), norm="forward")
        return x_out


def get_grid(B, H, W, device):
    y = torch.linspace(0, 1, steps=H, device=device)
    x = torch.linspace(0, 1, steps=W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    return grid


class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 modes1=12, modes2=12, width=64, depth=4,
                 add_grid=True, dropout=0.0):
        super().__init__()
        self.add_grid = add_grid
        lift_in = in_channels + (2 if add_grid else 0)
        self.fc0 = nn.Conv2d(lift_in, width, 1)
        self.spectral_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        self.bn = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(SpectralConv2d(width, width, modes1, modes2))
            self.w_layers.append(nn.Conv2d(width, width, 1))
            self.bn.append(nn.BatchNorm2d(width))
        self.fc1 = nn.Conv2d(width, width // 2, 1)
        self.fc2 = nn.Conv2d(width // 2, out_channels, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        B, _, H, W = x.shape
        if self.add_grid:
            grid = get_grid(B, H, W, x.device)
            x = torch.cat([x, grid], dim=1)
        x = self.fc0(x)
        for spec, wlin, bn in zip(self.spectral_layers, self.w_layers, self.bn):
            y = spec(x) + wlin(x)
            x = bn(F.gelu(y))
            x = self.dropout(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = get_device()
    print("Using device:", device)
    B, Cin, Cout, H, W = 2, 3, 1, 128, 128
    x = torch.randn(B, Cin, H, W, device=device)
    model = FNO2d(Cin, Cout, modes1=16, modes2=16, width=64, depth=4).to(device)
    with torch.no_grad():
        y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("#params     :", count_parameters(model))

