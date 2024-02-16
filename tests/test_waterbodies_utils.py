import torch

from datamodules.waterbodies_utils import im_mask_transform


def test_im_mask_transform():
    im = "tests/test_data/im1.jpg"
    mask = "tests/test_data/mask1.jpg"
    height, width = 512, 512
    im, mask = im_mask_transform(im, mask, height, width)
    assert im.shape == torch.Size([3, 512, 512])
    assert mask.shape == torch.Size([1, 512, 512])


if __name__ == "__main__":
    test_im_mask_transform()
