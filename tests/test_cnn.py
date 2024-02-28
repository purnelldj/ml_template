import torch

from models.cnn import (
    CNN2layer,
    CNN2layerBN,
    CNN3layer,
    CNN3layerLarge,
    CNN4layer,
    CNN4layerLarge,
)


def test_cnn():
    nets = [
        CNN2layer(),
        CNN2layerBN(),
        CNN3layer(),
        CNN4layer(),
        CNN3layerLarge(),
        CNN4layerLarge(),
    ]
    for net in nets:
        tt = torch.rand(1, 3, 64, 64)
        net = CNN2layer()
        logits = net(tt)
        assert logits.shape == torch.Size([1, 10])
        tt = torch.rand(100, 3, 64, 64)
        net = CNN2layer()
        logits = net(tt)
        assert logits.shape == torch.Size([100, 10])
