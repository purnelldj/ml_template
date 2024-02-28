import torch

from datamodules.EuroSAT_RGB_Samples_utils import file_to_class, file_to_im


def test_file_to_class():
    file = "EuroSAT_RGB_Samples/SeaLake_Samples/SeaLake_995.jpg"
    class_true = "SeaLake"
    class_test = file_to_class(file)
    assert class_true == class_test
    file = "EuroSAT_RGB_Samples/Forest_Samples/Forest_71.jpg"
    class_true = "Forest"
    class_test = file_to_class(file)
    assert class_true == class_test


def test_file_to_im():
    im = file_to_im("tests/test_data/AnnualCrop_155.jpg")
    assert im.shape == torch.Size([3, 64, 64])
    assert im.max().numpy() <= 1
    assert im.min().numpy() >= 0
    im = file_to_im("tests/test_data/River_143.jpg")
    assert im.shape == torch.Size([3, 64, 64])
    assert im.max().numpy() <= 1
    assert im.min().numpy() >= 0


if __name__ == "__main__":
    test_file_to_im()
