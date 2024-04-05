import torch

from models.cnn import CNN3layer
from models.resnet import ResNet13chanV1


def test_cnn():
    nets = [CNN3layer()]
    for net in nets:
        tt = torch.rand(1, 3, 64, 64)
        logits = net(tt)
        assert logits.shape == torch.Size([1, 10])
        tt = torch.rand(100, 3, 64, 64)
        logits = net(tt)
        assert logits.shape == torch.Size([100, 10])


def test_rn():
    params = ["net", "optimizer", "criterion", "accuracy"]
    kwargs = {param: None for param in params}
    nets = [ResNet13chanV1(**kwargs)]
    for net in nets:
        tt = torch.rand(1, 13, 224, 224)
        logits = net(tt)
        assert logits.shape == torch.Size([1, 10])
        tt = torch.rand(100, 13, 224, 224)
        logits = net(tt)
        assert logits.shape == torch.Size([100, 10])


if __name__ == "__main__":
    test_rn()
