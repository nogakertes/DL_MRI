import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def add_l2_reg(loss,net):
    """add l2 regularization to loss"""
    l2_lambda = 0.001
    l2_norm = sum(p.pow(2.0).sum()
                  for p in net.parameters())
    reg_loss = loss + l2_lambda * l2_norm
    return reg_loss

## all functions needed for calculation of SSIM score

def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

    # Converting to 2D
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def ssim(img1, img2, DEVICE, window_size=11):
    #L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),
    b,c,h,w = img1.shape
    pad = window_size // 2
    window = create_window(window_size=11,channel=c).to(DEVICE)
    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=c)
    mu2 = F.conv2d(img2, window, padding=pad, groups=c)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=c) - mu12
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    return ssim_score.mean()

#
# class SSIMLoss(nn.Module):
#     """
#     SSIM loss module.
#     """
#
#     def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
#         """
#         Args:
#             win_size: Window size for SSIM calculation.
#             k1: k1 parameter for SSIM calculation.
#             k2: k2 parameter for SSIM calculation.
#         """
#         super().__init__()
#         self.win_size = win_size
#         self.k1, self.k2 = k1, k2
#         self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
#         NP = win_size ** 2
#         self.cov_norm = NP / (NP - 1)
#
#     def forward(
#         self,
#         X: torch.Tensor,
#         Y: torch.Tensor,
#         DEVICE : str,
#         reduced: bool = True,
#     ):
#         S = ssim(X,Y,DEVICE)
#         if reduced:
#             return 1 - S.mean()
#         else:
#             return 1 - S

# def add_ssim_reg(loss,pred,y,DEVICE):
#     """add ssim regularization to loss"""
#     ssim_lambda = 0.001
#     reg_loss = loss + (1-ssim(pred,y).to(DEVICE)) * ssim_lambda
#     return reg_loss



class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 2, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        data_range: torch.Tensor,
        DEVICE: str,
        reduced: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)
        self.w = self.w.to(DEVICE)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced:
            return 1 - S.mean()
        else:
            return 1 - S
