import torch

from models.cnn import CNN3layer


def test_cnn():
    nets = [CNN3layer()]
    for net in nets:
        tt = torch.rand(1, 3, 64, 64)
        logits = net(tt)
        assert logits.shape == torch.Size([1, 10])
        tt = torch.rand(100, 3, 64, 64)
        logits = net(tt)
        assert logits.shape == torch.Size([100, 10])
