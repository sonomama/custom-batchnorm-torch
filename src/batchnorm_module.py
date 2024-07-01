import torch
import torch.nn as nn

from src.batchnorm_func import CustomBatchNormFunc


class CustomBatchNormModule(nn.Module):
    def __init__(self, in_features: int, eps: float = 1e-5):
        super(CustomBatchNormModule, self).__init__()
        self.in_features = in_features
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(self.in_features, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(self.in_features, dtype=torch.float32))

    def forward(self, input_tensor: torch.Tensor):
        if input_tensor.shape[1] != self.in_features:
            raise ValueError(
                f"Inpute tensor expected to have {self.in_features} features, "
                f"received {input_tensor.shape[1]}"
            )

        out = CustomBatchNormFunc.apply(input_tensor, self.gamma, self.beta, self.eps)
        return out
