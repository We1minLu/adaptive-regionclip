"""Adaptive Teacher appearance augmentation with isolated per-sample CPU RNG.

The transform parameters/order follow facebookresearch/adaptive_teacher,
commit 5256463ad9ec90fd5ba84ebb8d53bed56bd369df,
adapteacher/data/detection_utils.py and transforms/augmentation_impl.py.
Geometric transforms are shared by the two views in data.py. No resize/crop
is introduced here; the experiment retains its original image resolution.
"""
from contextlib import contextmanager
import random


class GaussianBlur:
    """The PIL-radius blur used by Adaptive Teacher (not a fixed kernel)."""

    def __init__(self, sigma=(0.1, 2.0)):
        self.sigma = sigma

    def __call__(self, image):
        from PIL import ImageFilter
        return image.filter(ImageFilter.GaussianBlur(
            radius=random.uniform(self.sigma[0], self.sigma[1])))


def build_strong_augmentation():
    """Keep torchvision's ColorJitter/RandomErasing sampling semantics intact."""
    from torchvision import transforms as T
    return T.Compose([
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianBlur((0.1, 2.0))], p=0.5),
        T.Compose([
            T.ToTensor(),
            T.RandomErasing(p=0.7, scale=(0.05, 0.2), ratio=(0.3, 3.3), value='random'),
            T.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.1, 6), value='random'),
            T.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.05, 8), value='random'),
            T.ToPILImage(),
        ]),
    ])


@contextmanager
def isolated_cpu_rng(seed):
    """Restore RNG even on failure; never initialize or seed CUDA in a worker.

    torchvision's transforms use the CPU torch RNG, while the reference PIL
    blur uses Python random. torch.manual_seed/fork_rng are deliberately not
    used because they can touch CUDA. DataLoader workers execute sequentially.
    """
    import torch
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(int(seed))
        torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(int(seed)).get_state())
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)


def apply_strong_augmentation(image, seed, transform):
    """Transform a true RGB PIL image without consuming caller RNG state."""
    if image.mode != 'RGB':
        raise ValueError('Strong augmentation requires an RGB PIL image')
    with isolated_cpu_rng(seed):
        return transform(image)
