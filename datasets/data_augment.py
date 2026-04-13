"""
Paired data augmentation transforms for low-light image enhancement.

All transforms apply identical operations to both input (low-light)
and label (normal-light) images to maintain correspondence.
"""

import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):
    """Random crop applied identically to both images."""

    def __call__(self, image, label):

        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            label = F.pad(label, (self.size[1] - label.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            label = F.pad(label, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)

        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    """Compose multiple paired transforms sequentially."""

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    """Random horizontal flip applied identically to both images."""

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(label)
        return img, label


class PairRandomVerticalFlip(transforms.RandomVerticalFlip):
    """Random vertical flip applied identically to both images."""

    def __call__(self, img, label):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(label)
        return img, label


class PairRandomRotation:
    """Random 90-degree rotation applied identically to both images.

    Randomly rotates both images by 0°, 90°, 180°, or 270°.
    This is geometrically exact (no interpolation artifacts).
    """

    def __call__(self, img, label):
        angle = random.choice([0, 90, 180, 270])
        if angle == 0:
            return img, label
        return F.rotate(img, angle), F.rotate(label, angle)


class PairColorJitter:
    """Mild brightness/contrast jitter applied only to the input (low-light) image.

    Note: This is intentionally applied only to the input, NOT the target,
    to simulate varying exposure conditions and improve robustness.

    Args:
        brightness: Brightness jitter range. Default: 0.1.
        contrast: Contrast jitter range. Default: 0.1.
    """

    def __init__(self, brightness=0.1, contrast=0.1):
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast)

    def __call__(self, img, label):
        # Only jitter the input, not the target
        img = self.jitter(img)
        return img, label


class PairToTensor(transforms.ToTensor):
    """Convert both PIL Images to tensors."""

    def __call__(self, pic, label):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic), F.to_tensor(label)
