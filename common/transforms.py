from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import monai.transforms as _mt


def ct_window(arr: np.ndarray, lo: float = -1000.0, hi: float = 1000.0) -> np.ndarray:
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo)).astype(np.float32)


def minmax_norm(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a, b = float(arr.min()), float(arr.max())
    return ((arr - a) / (b - a + eps)).astype(np.float32)


def us_scale255(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0.0, 255.0) / 255.0).astype(np.float32)


def us_zscore(arr):
    m = arr > 0
    if float(m.sum()) == 0:
        return arr
    vals = arr[m]
    mu = vals.mean()
    sd = ((vals - mu) ** 2).mean() ** 0.5
    out = arr.clone() if hasattr(arr, "clone") else arr.copy().astype(np.float32)
    out[m] = (vals - mu) / (sd + 1e-8)
    return out


def fourier_amp_dr(img: np.ndarray, beta: float = 0.15, eta=(0.5, 1.6)) -> np.ndarray:
    x = img.astype(np.float32)
    if x.ndim == 2:
        x = x[..., None]
    H, W, C = x.shape
    b = max(1, int(round(min(H, W) * beta / 2)))
    cy, cx = H // 2, W // 2
    y0, y1, x0, x1 = cy - b, cy + b, cx - b, cx + b
    field = np.random.uniform(eta[0], eta[1], size=(y1 - y0, x1 - x0)).astype(np.float32)
    out = np.empty_like(x)
    for c in range(C):
        f = np.fft.fftshift(np.fft.fft2(x[..., c]))
        amp, pha = np.abs(f), np.angle(f)
        amp[y0:y1, x0:x1] *= field
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha))))
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[..., 0] if img.ndim == 2 else out


def _fourier_lambda(image, **kwargs):
    return fourier_amp_dr(image)


def _specular_blobs(image, **kwargs):
    img = image.astype(np.float32)
    H, W = img.shape[:2]
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    alpha = np.zeros((H, W), np.float32)
    for _ in range(int(np.random.randint(1, 5))):
        cy, cx = np.random.uniform(0, H), np.random.uniform(0, W)
        sy, sx = np.random.uniform(0.02, 0.08) * H, np.random.uniform(0.02, 0.08) * W
        peak = np.random.uniform(0.5, 0.95)
        alpha = np.maximum(alpha, peak * np.exp(-(((ys - cy) ** 2) / (2 * sy * sy)
                                                  + ((xs - cx) ** 2) / (2 * sx * sx))))
    alpha = alpha[..., None]
    white = np.array([255.0, 250.0, 245.0], np.float32)
    return np.clip(img * (1.0 - alpha) + white * alpha, 0, 255).astype(np.uint8)


def _blood_pools(image, **kwargs):
    import cv2

    img = image.astype(np.float32)
    H, W = img.shape[:2]
    mask = np.zeros((H, W), np.float32)
    for _ in range(int(np.random.randint(1, 4))):
        cy, cx = int(np.random.randint(0, H)), int(np.random.randint(0, W))
        ay = int(np.random.randint(int(0.05 * H), max(int(0.25 * H), int(0.06 * H) + 1)))
        ax = int(np.random.randint(int(0.05 * W), max(int(0.25 * W), int(0.06 * W) + 1)))
        cv2.ellipse(mask, (cx, cy), (ax, ay), int(np.random.randint(0, 180)), 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(H, W) * 0.01) * np.random.uniform(0.25, 0.6)
    color = np.array([float(np.random.randint(90, 160)), float(np.random.randint(5, 35)),
                      float(np.random.randint(5, 35))], np.float32)
    m = mask[..., None]
    return np.clip(img * (1.0 - m) + color * m, 0, 255).astype(np.uint8)


def _specular_lambda(image, **kwargs):
    return _specular_blobs(image)


def _blood_lambda(image, **kwargs):
    return _blood_pools(image)


def _norm_3d(modality, hu_clip):
    if modality == "ct":
        return _mt.ScaleIntensityRanged(keys="image", a_min=hu_clip[0], a_max=hu_clip[1],
                                        b_min=0.0, b_max=1.0, clip=True)
    return _mt.Lambdad(keys="image", func=us_zscore)


def build_3d_train_transforms(patch_size: Sequence[int], modality: str = "ct",
                              hu_clip: Tuple[float, float] = (-1000, 1000),
                              oversample_fg: float = 0.33, speckle: bool = False,
                              boundary_sdf_classes: int = 0, sampling=None, heavy: bool = False,
                              spacing_aug: bool = False):
    T = _mt
    aug = [
        T.RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        T.RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        T.RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
        T.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        T.RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        T.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        T.RandGaussianNoised(keys="image", prob=0.15, std=0.02),
    ]
    if speckle or heavy:
        aug += [
            T.RandAffined(keys=["image", "label"], prob=0.3,
                          rotate_range=(0.26, 0.26, 0.26), scale_range=(0.15, 0.15, 0.15),
                          mode=("bilinear", "nearest"), padding_mode="zeros"),
            T.RandGaussianSmoothd(keys="image", prob=0.2,
                                  sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
            T.RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.7, 1.5)),
        ]
    if heavy:
        aug += [
            T.RandScaleIntensityd(keys="image", factors=0.3, prob=0.4),
            T.RandShiftIntensityd(keys="image", offsets=0.15, prob=0.4),
            RandFBPKerneld(keys="image", prob=0.35),
            T.RandAdjustContrastd(keys="image", prob=0.3, gamma=(0.6, 1.7)),
        ]
    if spacing_aug:
        aug += [
            T.RandSimulateLowResolutiond(keys="image", prob=0.25, zoom_range=(0.5, 1.0)),
            T.RandZoomd(keys=["image", "label"], prob=0.2, min_zoom=(0.85, 0.85, 0.7),
                        max_zoom=(1.15, 1.15, 1.4), mode=("trilinear", "nearest"), keep_size=True),
        ]
    if speckle:
        aug += [
            T.Rand3DElasticd(keys=["image", "label"], prob=0.2, sigma_range=(5, 8),
                             magnitude_range=(50, 120), mode=("bilinear", "nearest")),
            _SpeckleNoised(keys="image", prob=0.2, var=0.05),
        ]
    return T.Compose([
        T.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        T.EnsureTyped(keys=["image", "label"]),
        _norm_3d(modality, hu_clip),
        T.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        _ClampLabeld("label", "label_pos"),
        T.RandCropByPosNegLabeld(keys=["image", "label", "label_pos"], label_key="label_pos",
                                 spatial_size=tuple(patch_size), pos=oversample_fg,
                                 neg=1 - oversample_fg, num_samples=1, allow_smaller=True),
        T.DeleteItemsd(keys=["label_pos"]),
        T.SpatialPadd(keys=["image", "label"], spatial_size=tuple(patch_size)),
        *aug,
        *([_BoundarySDFd("label", boundary_sdf_classes, sampling)] if boundary_sdf_classes else []),
        T.EnsureTyped(keys=["image", "label", "dist_maps"], allow_missing_keys=True, track_meta=False),
    ])


def build_3d_val_transforms(modality: str = "ct", hu_clip: Tuple[float, float] = (-1000, 1000)):
    T = _mt
    return T.Compose([
        T.LoadImaged(keys=["image", "label"], ensure_channel_first=True, allow_missing_keys=True),
        T.EnsureTyped(keys=["image", "label"], allow_missing_keys=True),
        _norm_3d(modality, hu_clip),
    ])


class _ClampLabeld(_mt.MapTransform):
    def __init__(self, src="label", dst="label_pos"):
        _mt.MapTransform.__init__(self, [src])
        self.src, self.dst = src, dst

    def __call__(self, data):
        d = dict(data)
        lbl = d[self.src]
        d[self.dst] = lbl.clamp(min=0) if hasattr(lbl, "clamp") else np.clip(lbl, 0, None)
        return d


class _BoundarySDFd(_mt.MapTransform):
    def __init__(self, label_key, num_classes, sampling=None):
        _mt.MapTransform.__init__(self, [label_key])
        self.label_key, self.num_classes, self.sampling = label_key, num_classes, sampling

    def _per_case_spacing(self, lbl):
        aff = getattr(lbl, "affine", None)
        if aff is not None:
            a = aff.detach().cpu().numpy() if hasattr(aff, "detach") else np.asarray(aff)
            if a.shape == (4, 4):
                sp = np.linalg.norm(a[:3, :3], axis=0)
                if np.all(np.isfinite(sp)) and np.all(sp > 0):
                    return tuple(float(x) for x in sp)
        return self.sampling

    def __call__(self, data):
        import torch

        from common.sdf import mask_to_sdf

        d = dict(data)
        lbl = d[self.label_key]
        sampling = self._per_case_spacing(lbl)
        arr = lbl.detach().cpu().numpy() if hasattr(lbl, "detach") else np.asarray(lbl)
        if arr.ndim == 4:
            arr = arr[0]
        chans = np.stack([mask_to_sdf(arr == c, sampling, normalize=False)
                          for c in range(1, max(self.num_classes, 2))], 0).astype(np.float32)
        chans[:, arr < 0] = 0.0
        d["dist_maps"] = torch.from_numpy(chans)
        return d


class _SpeckleNoised(_mt.RandomizableTransform, _mt.MapTransform):
    def __init__(self, keys, prob=0.2, var=0.05):
        _mt.MapTransform.__init__(self, keys)
        _mt.RandomizableTransform.__init__(self, prob)
        self.var = var

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d
        for k in self.keys:
            img = d[k]
            noise = np.random.normal(0.0, self.var ** 0.5, size=tuple(img.shape)).astype(np.float32)
            d[k] = img * (1.0 + noise)
        return d


def rand_conv(x: "torch.Tensor", p: float = 0.5, kernel_sizes: Sequence[int] = (1, 3, 5, 7),
              mix: bool = True) -> "torch.Tensor":
    if float(torch.rand(())) > p:
        return x
    B, C, H, W = x.shape
    k = int(kernel_sizes[int(torch.randint(len(kernel_sizes), ()))])
    w = torch.randn(C, C, k, k, device=x.device, dtype=x.dtype) * (2.0 / (C * k * k)) ** 0.5
    y = F.conv2d(x, w, padding=k // 2)
    xm = x.mean(dim=(2, 3), keepdim=True); xs = x.std(dim=(2, 3), keepdim=True) + 1e-5
    ym = y.mean(dim=(2, 3), keepdim=True); ys = y.std(dim=(2, 3), keepdim=True) + 1e-5
    y = (y - ym) / ys * xs + xm
    if mix:
        a = torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype)
        y = a * x + (1.0 - a) * y
    return y.clamp(0.0, 1.0)


_FBP_RHO_CACHE = {}


def _fbp_radial(shape):
    if shape not in _FBP_RHO_CACHE:
        freqs = [torch.fft.fftfreq(s) for s in shape]
        grid = torch.meshgrid(*freqs, indexing="ij")
        rho = torch.sqrt(sum(g ** 2 for g in grid))
        _FBP_RHO_CACHE[shape] = rho / (rho.max() + 1e-8)
    return _FBP_RHO_CACHE[shape]


class RandFBPKerneld(_mt.RandomizableTransform, _mt.MapTransform):
    def __init__(self, keys, prob=0.35, alpha=(-0.6, 1.2), beta=(1.5, 3.0), allow_missing_keys=False):
        _mt.MapTransform.__init__(self, keys, allow_missing_keys)
        _mt.RandomizableTransform.__init__(self, prob)
        self.alpha, self.beta = alpha, beta

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d
        a = float(self.R.uniform(*self.alpha))
        b = float(self.R.uniform(*self.beta))
        for key in self.key_iterator(d):
            img = d[key]
            vol = img[0].float()
            gain = (1.0 + a * _fbp_radial(tuple(vol.shape)).pow(b)).clamp(0.2, 3.0)
            img[0] = torch.fft.ifftn(torch.fft.fftn(vol) * gain).real.to(img.dtype)
        return d


def tensor_fbpaug(x: "torch.Tensor", alpha=(-0.6, 1.2), beta=(1.5, 3.0)) -> "torch.Tensor":
    spatial = tuple(x.shape[2:])
    rho = _fbp_radial(spatial).to(x.device)
    dims = tuple(range(2, x.ndim))
    gains = []
    for _ in range(x.shape[0]):
        a = float(np.random.uniform(*alpha))
        b = float(np.random.uniform(*beta))
        gains.append((1.0 + a * rho.pow(b)).clamp(0.2, 3.0))
    gain = torch.stack(gains).unsqueeze(1)
    f = torch.fft.fftn(x.float(), dim=dims)
    return torch.fft.ifftn(f * gain, dim=dims).real.to(x.dtype)


_WEAK_2D = {}
_STRONG_2D = None


def build_2d_train_transforms(img_size: Sequence[int], heavy: bool = False,
                              surg_nuisance: bool = False, nuisance_p: float = 0.3):
    import albumentations as A

    key = (tuple(img_size), heavy, surg_nuisance, round(float(nuisance_p), 3))
    if key not in _WEAK_2D:
        if heavy:
            heavy_pre = [
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=20,
                                   border_mode=0, p=0.7),
                A.OneOf([
                    A.ColorJitter(0.5, 0.5, 0.5, 0.15, p=1.0),
                    A.HueSaturationValue(25, 40, 30, p=1.0),
                    A.RGBShift(25, 25, 25, p=1.0),
                ], p=0.9),
                A.RandomBrightnessContrast(0.35, 0.35, p=0.7),
                A.RandomGamma((70, 150), p=0.4),
                A.CLAHE(clip_limit=3.0, p=0.3),
            ]
            nuisance = [
                A.Lambda(image=_specular_lambda, name="specular", p=nuisance_p),
                A.Lambda(image=_blood_lambda, name="blood", p=nuisance_p),
            ] if surg_nuisance else []
            heavy_post = [
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                    A.MotionBlur(blur_limit=9, p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.4),
                A.OneOf([A.GaussNoise(p=1.0), A.ISONoise(p=1.0)], p=0.4),
                A.RandomFog(fog_coef_range=(0.05, 0.3), alpha_coef=0.08, p=0.2),
                A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.04, 0.09),
                                hole_width_range=(0.04, 0.09), fill=0, p=0.2),
                A.Lambda(image=_fourier_lambda, name="fourier_dr", p=0.4),
            ]
            _WEAK_2D[key] = A.Compose(heavy_pre + nuisance + heavy_post)
        else:
            _WEAK_2D[key] = A.Compose([
                A.Resize(img_size[0], img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.3),
            ])
    tf = _WEAK_2D[key]

    def _apply(img, mask):
        out = tf(image=img, mask=mask)
        return out["image"], out["mask"]

    return _apply


def strong_augment_2d(img: np.ndarray) -> np.ndarray:
    import albumentations as A

    global _STRONG_2D
    if _STRONG_2D is None:
        _STRONG_2D = A.Compose([
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.GaussNoise(p=0.3),
            A.Lambda(image=_fourier_lambda, name="fourier_dr", p=0.4),
        ])
    return _STRONG_2D(image=img)["image"]
