import torch

from datamodules.eurosat_rgb_utils import EuTransform, ViTransform, file_to_class


def test_file_to_class():
    file = "eurosat_rgb/SeaLake_Samples/SeaLake_995.jpg"
    class_true = "SeaLake"
    class_test = file_to_class(file)
    assert class_true == class_test
    file = "eurosat_rgb/Forest_Samples/Forest_71.jpg"
    class_true = "Forest"
    class_test = file_to_class(file)
    assert class_true == class_test


def test_file_to_im():
    eutfm = EuTransform(im_height=224, im_width=224)
    im, label = eutfm("tests/test_data/AnnualCrop_155.jpg")
    assert im.shape == torch.Size([3, 224, 224])
    assert im.max().numpy() <= 1
    assert im.min().numpy() >= 0
    assert label == 0
    vitfm = ViTransform()
    im, label = vitfm("tests/test_data/River_143.jpg")
    assert im.shape == torch.Size([3, 224, 224])
    assert label == 8
    vitfm = ViTransform()


if __name__ == "__main__":
    test_file_to_im()
