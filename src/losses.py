import torch
import torch.nn.functional as F

def grad_xy(t):
    # central difference approximation
    dx = t[..., :, 2:] - t[..., :, :-2]
    dy = t[..., 2:, :] - t[..., :-2, :]
    # pad back to original size
    dx = F.pad(dx, (1,1,0,0))
    dy = F.pad(dy, (0,0,1,1))
    return dx, dy

def surrogate_loss(pred, tgt, w_mse=1.0, w_grad=0.05, w_div=0.05):
    # pred, tgt: (B,3,H,W)
    mse = F.mse_loss(pred, tgt)

    # gradient matching (all channels)
    gx_p, gy_p = grad_xy(pred)
    gx_t, gy_t = grad_xy(tgt)
    grad_l = F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)

    # divergence penalty (on u,v only)
    u, v = pred[:,0:1], pred[:,1:2]
    dux, _ = grad_xy(u)
    _, dvy = grad_xy(v)
    div = dux + dvy
    div_l = (div**2).mean()

    total = w_mse*mse + w_grad*grad_l + w_div*div_l
    return total, {"mse": mse.item(), "grad": grad_l.item(), "div": div_l.item()}

