import torch
from torchvision.transforms import functional as F, transforms as T, InterpolationMode
from torch import nn, Tensor


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Tensor, target=None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target


class PILToTensor(nn.Module):
    def forward(self, image: Tensor, target=None):
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, image: Tensor, target=None):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class ScaleJitter(nn.Module):
    """Randomly resizes the image and its bounding boxes  within the specified scale range.
    The class implements the Scale Jitter augmentation as described in the paper
    `"Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation" <https://arxiv.org/abs/2012.07177>`_.
    Args:
        target_size (tuple of ints): The target size for the transform provided in (height, weight) format.
        scale_range (tuple of ints): scaling factor interval, e.g (a, b), then scale is randomly sampled from the
            range a <= scale <= b.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
    """

    def __init__(
        self,
        target_size,
        scale_range=(0.5, 1.5),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    ):
        super().__init__()
        self.target_size = target_size
        self.scale_range = scale_range
        self.interpolation = interpolation

    def forward(self, image: Tensor, target=None):
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError(
                    f"image should be 2/3 dimensional. Got {image.ndimension()} dimensions."
                )
            elif image.ndimension() == 2:
                image = image.unsqueeze(0)

        _, orig_height, orig_width = F.get_dimensions(image)

        scale = self.scale_range[0] + torch.rand(1) * (
            self.scale_range[1] - self.scale_range[0]
        )
        r = (
            min(self.target_size[1] / orig_height, self.target_size[0] / orig_width)
            * scale
        )
        new_width = int(orig_width * r)
        new_height = int(orig_height * r)

        image = F.resize(
            image, [new_height, new_width], interpolation=self.interpolation
        )

        if target is not None:
            target["boxes"][:, 0::2] *= new_width / orig_width
            target["boxes"][:, 1::2] *= new_height / orig_height

        return image, target


class FixedSizeCrop(nn.Module):
    def __init__(self, size, fill=0, padding_mode="constant"):
        super().__init__()
        size = tuple(
            T._setup_size(
                size, error_msg="Please provide only two dimensions (w, h) for size."
            )
        )
        self.crop_height = size[1]
        self.crop_width = size[0]
        self.fill = (
            fill  # TODO: Fill is currently respected only on PIL. Apply tensor patch.
        )
        self.padding_mode = padding_mode

    def _pad(self, img, target, padding):
        # Taken from the functional_tensor.py pad
        if isinstance(padding, int):
            pad_left = pad_right = pad_top = pad_bottom = padding
        elif len(padding) == 1:
            pad_left = pad_right = pad_top = pad_bottom = padding[0]
        elif len(padding) == 2:
            pad_left = pad_right = padding[0]
            pad_top = pad_bottom = padding[1]
        else:
            pad_left = padding[0]
            pad_top = padding[1]
            pad_right = padding[2]
            pad_bottom = padding[3]

        padding = [pad_left, pad_top, pad_right, pad_bottom]
        img = F.pad(img, padding, self.fill, self.padding_mode)
        if target is not None:
            target["boxes"][:, 0::2] += pad_left
            target["boxes"][:, 1::2] += pad_top
            if "masks" in target:
                target["masks"] = F.pad(target["masks"], padding, 0, "constant")

        return img, target

    def _crop(self, img, target, top, left, height, width):
        img = F.crop(img, top, left, height, width)
        if target is not None:
            boxes = target["boxes"]
            boxes[:, 0::2] -= left
            boxes[:, 1::2] -= top
            boxes[:, 0::2].clamp_(min=0, max=width)
            boxes[:, 1::2].clamp_(min=0, max=height)

            is_valid = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])

            target["boxes"] = boxes[is_valid]
            target["labels"] = target["labels"][is_valid]
            if "masks" in target:
                target["masks"] = F.crop(
                    target["masks"][is_valid], top, left, height, width
                )

        return img, target

    def forward(self, img, target=None):
        _, height, width = F.get_dimensions(img)
        new_height = min(height, self.crop_height)
        new_width = min(width, self.crop_width)

        if new_height != height or new_width != width:
            offset_height = max(height - self.crop_height, 0)
            offset_width = max(width - self.crop_width, 0)

            r = torch.rand(1)
            top = int(offset_height * r)
            left = int(offset_width * r)

            img, target = self._crop(img, target, top, left, new_height, new_width)

        pad_bottom = max(self.crop_height - new_height, 0)
        pad_right = max(self.crop_width - new_width, 0)
        if pad_bottom != 0 or pad_right != 0:
            img, target = self._pad(img, target, [0, 0, pad_right, pad_bottom])

        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(PILToTensor())
    transforms.append(ConvertImageDtype(torch.float))
    if train:
        transforms.append(ScaleJitter(target_size=(1300, 725)))
        transforms.append(FixedSizeCrop(size=(1300, 725)))
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)
