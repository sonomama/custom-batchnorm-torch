import torch
import sys
sys.path.append("..")
from src.batchnorm_func import CustomBatchNormFunc


def test_bn_backprop_works_as_pytorch():
    x = torch.tensor([[10.111, -3.15, 5.365], [-100.982, 1.456, 93.019], [2.543, 8.582, 15.675], [1.354, 8.23, 4.22], [1.04, -6.365, 15.291]], requires_grad=True, dtype=torch.float32)
    gamma = torch.tensor([1.0, 1.0, 1.0], requires_grad=True, dtype=torch.float32)
    beta = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, dtype=torch.float32)
    loss_fn = lambda t: torch.sum(t.abs())
    f = CustomBatchNormFunc.apply
    cbn = f(x, gamma, beta)
    loss = loss_fn(cbn)
    loss.backward()

    actual_x_grad = x.grad
    actual_gamma_grad = gamma.grad
    actual_beta_grad = beta.grad

    bn = torch.nn.BatchNorm1d(x.shape[1], dtype=torch.float32)
    xx = x.clone().detach()
    xx.requires_grad = True
    out = bn(xx)
    loss = loss_fn(out)
    loss.backward()

    assert torch.allclose(actual_x_grad, xx.grad)
    assert torch.allclose(actual_gamma_grad, bn.weight.grad)
    assert torch.allclose(actual_beta_grad, bn.bias.grad)