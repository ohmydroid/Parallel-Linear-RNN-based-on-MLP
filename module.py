import torch
import torch.nn as nn


class HardSigmoid(nn.Module):
      def __init__(self, inplace=True):
          super(HardSigmoid, self).__init__()
          self.relu = nn.ReLU6(inplace=inplace)

      def forward(self, x):
          return self.relu(x + 3) / 6 - 0.5


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LinearRNNBlock(nn.Module):

      def __init__(self, channel, expansion=1):
          super(LinearRNNBlock, self).__init__()
          self.norm1 = RMSNorm(channel)
          self.mlp1 = nn.Linear(channel, channel)
          
          self.norm2 = RMSNorm(channel)
          self.mlp2 = nn.Sequential(nn.Linear(channel, expansion*channel),
                                    nn.ReLU(True),
                                    nn.Linear(expansion*channel, channel))

          self.act = HardSigmoid(True)
          
          
      def forward(self, x):
  
          T = x.size(-2)
          state  = self.act(self.norm1(x)).cumsum(dim=1)
          out = self.act(self.mlp1(state))*x
          out = self.mlp2(self.norm2(out)) + out
          return out
