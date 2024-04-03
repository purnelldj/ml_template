import torch

from datamodules.waterbodies_utils import WbTransform


def test_im_mask_transform():
    im = "tests/test_data/im1.jpg"
    mask = "tests/test_data/mask1.jpg"
    tfm = WbTransform(512, 512)
    im, mask = tfm(im, mask)
    assert im.shape == torch.Size([3, 512, 512])
    assert mask.shape == torch.Size([1, 512, 512])
    assert im.dtype == torch.float32
    assert mask.dtype == torch.float32


if __name__ == "__main__":
    test_im_mask_transform()
