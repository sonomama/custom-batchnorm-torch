import pytest
import torch
from src.batchnorm_func import CustomBatchNormFunc
from src.batchnorm_module import CustomBatchNormModule


@pytest.fixture
def input_tensor():
    return torch.tensor(
        [
            [10.111, -3.15, 5.365],
            [-100.982, 1.456, 93.019],
            [2.543, 8.582, 15.675],
            [1.354, 8.23, 4.22],
            [1.04, -6.365, 15.291],
        ],
        requires_grad=True,
        dtype=torch.float32,
    )


def loss_fn(t):
    return torch.sum(t.abs())


def test_bn_backprop_works_as_pytorch(input_tensor):
    gamma = torch.tensor([1.0, 1.0, 1.0], requires_grad=True, dtype=torch.float32)
    beta = torch.tensor([0.0, 0.0, 0.0], requires_grad=True, dtype=torch.float32)
    f = CustomBatchNormFunc.apply
    cbn = f(input_tensor, gamma, beta)
    loss = loss_fn(cbn)
    loss.backward()

    actual_x_grad = input_tensor.grad
    actual_gamma_grad = gamma.grad
    actual_beta_grad = beta.grad

    bn = torch.nn.BatchNorm1d(input_tensor.shape[1], dtype=torch.float32)
    xx = input_tensor.clone().detach()
    xx.requires_grad = True
    out = bn(xx)
    loss = loss_fn(out)
    loss.backward()

    assert torch.allclose(actual_x_grad, xx.grad)
    assert torch.allclose(actual_gamma_grad, bn.weight.grad)
    assert torch.allclose(actual_beta_grad, bn.bias.grad)


def test_can_apply_bn_module(input_tensor):
    my_bn_layer = CustomBatchNormModule(in_features=input_tensor.shape[1], eps=1e-5)
    custom_bn = my_bn_layer(input_tensor)
    loss = loss_fn(custom_bn)
    loss.backward()

    actual_x_grad = input_tensor.grad
    actual_gamma_grad = my_bn_layer.gamma.grad
    actual_beta_grad = my_bn_layer.beta.grad

    bn = torch.nn.BatchNorm1d(input_tensor.shape[1], dtype=torch.float32)
    xx = input_tensor.clone().detach()
    xx.requires_grad = True
    out = bn(xx)
    loss = loss_fn(out)
    loss.backward()

    assert torch.allclose(actual_x_grad, xx.grad)
    assert torch.allclose(actual_gamma_grad, bn.weight.grad)
    assert torch.allclose(actual_beta_grad, bn.bias.grad)
