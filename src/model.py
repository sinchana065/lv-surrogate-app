import torch
import torch.nn as nn
import torch.nn.functional as F

def CBR(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.block = nn.Sequential(CBR(c_in,c_out), CBR(c_out,c_out))
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        y = self.block(x)
        return self.pool(y), y

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in//2, 2, stride=2)
        self.block = nn.Sequential(CBR(c_in, c_out), CBR(c_out, c_out))
    def forward(self, x, skip):
        x = self.up(x)
        # pad if needed
        dh = skip.size(-2) - x.size(-2)
        dw = skip.size(-1) - x.size(-1)
        x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=6, out_ch=3, base=32):
        super().__init__()
        self.d1 = Down(in_ch, base)
        self.d2 = Down(base, base*2)
        self.d3 = Down(base*2, base*4)
        self.mid = nn.Sequential(CBR(base*4, base*8), CBR(base*8, base*8))
        self.u3 = Up(base*8, base*4)
        self.u2 = Up(base*4, base*2)
        self.u1 = Up(base*2, base)
        self.head = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        x1p, x1 = self.d1(x)
        x2p, x2 = self.d2(x1p)
        x3p, x3 = self.d3(x2p)
        m = self.mid(x3p)
        x = self.u3(m, x3)
        x = self.u2(x, x2)
        x = self.u1(x, x1)
        return self.head(x)

