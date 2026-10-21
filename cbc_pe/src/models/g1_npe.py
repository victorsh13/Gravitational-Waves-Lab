"""One joint X-only conditional flow; torch 1.9-compatible, no flow dependency."""
import math
import torch
from torch import nn

LABELS = ['chirp_mass', 'total_mass', 'chi_eff']


def mass_bounds(total):
    delta = torch.minimum(total - 10., 180. - total)
    upper = total * (0.25 ** 0.6)
    lower = total * ((1. - (delta / total) ** 2) / 4.).pow(0.6)
    return lower, upper


def to_unconstrained(y):
    """Physical -> logits, including exact triangular Jacobian.

    Stored float32 chirp masses can round onto/outside the equal-mass boundary.
    Accept at most four float32 ULPs; move those boundary coordinates inward.
    This is a declared quantized-label approximation, not exact discrete NLL.
    """
    y = y.double()
    c, t, s = y.unbind(-1)
    lo, hi = mass_bounds(t)
    width = hi - lo
    tol = 4 * torch.finfo(torch.float32).eps * c.abs().clamp(min=1)
    if not bool(((t > 10) & (t < 180) & (width > 0) &
                 (s > -1) & (s < 1) & (c >= lo-tol) & (c <= hi+tol)).all()):
        raise ValueError('Target outside G1 support (no rows discarded)')
    u0 = (c-lo)/width
    boundary = ((u0 <= 1e-8) | (u0 >= 1-1e-8))
    u = torch.stack((u0.clamp(1e-8, 1-1e-8), (t-10)/170, (s+1)/2), -1)
    z = torch.log(u) - torch.log1p(-u)
    logdet = -(torch.log(u) + torch.log1p(-u)).sum(-1) - torch.log(width*340)
    return z, logdet, boundary


def to_physical(z):
    u = torch.sigmoid(z.double())
    t = 10 + 170*u[..., 1]
    lo, hi = mass_bounds(t)
    return torch.stack((lo+(hi-lo)*u[..., 0], t, 2*u[..., 2]-1), -1)


class Coupling(nn.Module):
    def __init__(self, i, context=128):
        super().__init__()
        self.register_buffer('mask', torch.tensor([float(j == i % 3) for j in range(3)]))
        self.net = nn.Sequential(nn.Linear(context+3, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, context, inverse=False):
        fixed = x*self.mask
        shift, scale = self.net(torch.cat((fixed, context), -1)).chunk(2, -1)
        scale = 2*torch.tanh(scale)*(1-self.mask)
        shift = shift*(1-self.mask)
        out = (x-shift)*torch.exp(-scale) if inverse else x*torch.exp(scale)+shift
        return fixed + (1-self.mask)*out, (-scale if inverse else scale).sum(-1)


class G1Posterior(nn.Module):
    def __init__(self):
        super().__init__()
        layers, previous = [], 3
        for channels in (32, 48, 64, 96, 128):
            layers += [nn.Conv1d(previous, channels, 9, stride=4, padding=4),
                       nn.GroupNorm(8, channels), nn.ELU()]
            previous = channels
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                                     nn.Linear(1024, 128), nn.ELU())
        self.flow = nn.ModuleList([Coupling(i) for i in range(9)])

    def encode(self, x):
        if tuple(x.shape[1:]) != (3, 16384) or not bool(torch.isfinite(x).all()):
            raise ValueError('Expected finite frozen X (B,3,16384)')
        return self.encoder(x)

    def log_prob(self, y, context):
        z, jac, boundary = to_unconstrained(y)
        z = z.float()
        ld = torch.zeros_like(z[..., 0])
        for layer in self.flow:
            z, d = layer(z, context)
            ld = ld+d
        lp = -.5*(z*z+math.log(2*math.pi)).sum(-1)+ld+jac
        if not bool(torch.isfinite(lp).all()):
            raise ValueError('Nonfinite conditional density')
        return lp, boundary

    def sample(self, context, n):
        b = len(context)
        ctx = context[:, None, :].expand(b, n, 128).reshape(-1, 128)
        z = torch.randn(b*n, 3, device=context.device)
        for layer in reversed(self.flow):
            z, _ = layer(z, ctx, inverse=True)
        result = to_physical(z).reshape(b, n, 3)
        if not bool(torch.isfinite(result).all()):
            raise ValueError('Nonfinite posterior draws')
        return result
