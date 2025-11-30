
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

IMG_SIZE = 256
PATCH = 128
device = "cuda" if torch.cuda.is_available() else "cpu"

class ResidualBlock(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(c, c, 3, padding=1)
        )

    def forward(self, x):
        return x + self.b(x)

class ResidualInterpCNN(nn.Module):
    def __init__(self, blocks=5):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(True)
        )
        self.res = nn.Sequential(*[ResidualBlock(64) for _ in range(blocks)])
        self.exit = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        return self.exit(self.res(self.entry(x)))

# model = ResidualInterpCNN().to(device)


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class SRGAN_G(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(True)
        )
        self.res = nn.Sequential(*[ResBlock() for _ in range(8)])
        self.final = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, prev, nxt):
        x = torch.cat([prev, nxt], dim=1)
        return self.final(self.res(self.pre(x)))
class SRGAN_D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(128*(PATCH//4)*(PATCH//4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def sinusoidal_timestep_embedding(timesteps, dim):
    """
    Standard sinusoidal time embedding (as in DDPM, DDIM, LDM).
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=device) / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, time_dim=256):
        super().__init__()

        # better time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(True),
            nn.Linear(time_dim, time_dim),
        )

        self.inc = DoubleConv(in_ch + time_dim, base_ch)
        self.down1 = DoubleConv(base_ch, base_ch * 2)
        self.down2 = DoubleConv(base_ch * 2, base_ch * 4)

        self.up2   = DoubleConv(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up1   = DoubleConv(base_ch * 2 + base_ch, base_ch)

        self.outc  = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x, t):
        # sinusoidal → MLP → broadcast
        t_emb = sinusoidal_timestep_embedding(t, 256)
        t_emb = self.time_mlp(t_emb)
        t_emb = t_emb[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        # concat time into channels
        x = torch.cat([x, t_emb], dim=1)

        c1 = self.inc(x)
        c2 = self.down1(F.max_pool2d(c1, 2))
        c3 = self.down2(F.max_pool2d(c2, 2))

        u2 = F.interpolate(c3, scale_factor=2)
        u2 = self.up2(torch.cat([u2, c2], dim=1))

        u1 = F.interpolate(u2, scale_factor=2)
        u1 = self.up1(torch.cat([u1, c1], dim=1))

        return self.outc(u1)

class FastNoiseScheduler:
    def __init__(self, device, T=10):
        self.T = T
        self.device = device

        # load 1000-step DDPM scheduler (linear β)
        beta = torch.linspace(1e-4, 0.02, 1000)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, 0)

        # non-uniform sampling: 40% early, 60% late
        boundary = 699
        late_steps = int(T * 0.6)
        early_steps = T - late_steps

        idx_early = torch.linspace(0, boundary, early_steps).long()
        idx_late = torch.linspace(boundary, 999, late_steps).long()

        idxs = torch.sort(torch.cat([idx_early, idx_late]))[0]

        self.beta = beta[idxs].to(device)
        self.alpha = alpha[idxs].to(device)
        self.alpha_bar = alpha_bar[idxs].to(device)
class FastDDPM(nn.Module):
    def __init__(self, device, T=10):
        super().__init__()
        self.scheduler = FastNoiseScheduler(device, T)
        self.unet = UNet2D(in_ch=3).to(device)

    def forward(self, cond, target, t):
        noise = torch.randn_like(target)
        a_bar = self.scheduler.alpha_bar[t].view(-1,1,1,1)
        x_t = torch.sqrt(a_bar) * target + torch.sqrt(1 - a_bar) * noise

        pred_noise = self.unet(torch.cat([x_t, cond], dim=1), t)
        return F.mse_loss(pred_noise, noise)

@torch.no_grad()
def ddim_sample(model, prev, nxt, device):
    model.eval()

    cond = torch.cat([prev, nxt], 1)
    B,_,H,W = cond.shape
    x = torch.randn(B, 1, H, W).to(device)

    T = model.scheduler.T

    for i in reversed(range(T)):
        t = torch.full((B,), i, device=device, dtype=torch.long)

        eps = model.unet(torch.cat([x, cond], 1), t)

        a_bar = model.scheduler.alpha_bar[i]
        a_bar_prev = model.scheduler.alpha_bar[i-1] if i > 0 else torch.tensor(1.0).to(device)

        x0 = (x - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar)

        x = torch.sqrt(a_bar_prev) * x0 + torch.sqrt(1 - a_bar_prev) * eps

    return x.clamp(-1, 1)
