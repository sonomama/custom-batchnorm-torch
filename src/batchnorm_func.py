from typing import Tuple
import torch


class CustomBatchNormFunc(torch.autograd.Function):
    """
    Define forward and backward methods for a manual batch norm implementation.
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input_tensor, gamma, beta, eps=1e-5) -> torch.Tensor:
        mu = input_tensor.mean(dim=0)
        var = input_tensor.var(dim=0, unbiased=False)
        input_hat = (input_tensor - mu) / (var + eps).sqrt()

        out = gamma * input_hat + beta
        ctx.save_for_backward(input_hat, gamma, mu, var)
        ctx.eps = eps

        return out

    @staticmethod
    def backward(
        ctx, grad_output
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, type(None)]:
        """
        Returns grads for the same tensors as those passed in the forward method.
        Need to return None for the eps correction.
        """
        input_hat, gamma, mu, var = ctx.saved_tensors
        denom = 1 / (var + ctx.eps)
        batch_size = input_hat.shape[0]  # n
        dinput_hat = grad_output * gamma  # [n, d]
        dsigma_sq = -torch.sum(dinput_hat * input_hat, dim=0) * denom / 2  # [d]
        dmu = (
            -torch.sum(dinput_hat * denom ** (0.5), dim=0)
            - 2 * dsigma_sq * torch.sum(input_hat * denom ** (-0.5), dim=0) / batch_size
        )  # [d]
        dinput = (
            dinput_hat * denom ** (0.5)
            + dsigma_sq * 2 * input_hat * denom ** (-0.5) / batch_size
            + dmu / batch_size
        )

        dgamma = torch.sum(grad_output * input_hat, dim=0)
        dbeta = torch.sum(grad_output, dim=0)

        return dinput, dgamma, dbeta, None
