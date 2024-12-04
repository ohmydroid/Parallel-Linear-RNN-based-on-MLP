import torch
import torch.nn as nn

'''
class HardSigmoid(nn.Module):
      def __init__(self, inplace=True):
          super(HardSigmoid, self).__init__()
          self.relu = nn.ReLU6(inplace=inplace)

      def forward(self, x):
          return self.relu(x + 3) / 6
'''

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

      def __init__(self, channel):
          super(LinearRNNBlock, self).__init__()
          self.norm1 = RMSNorm(channel, expansion=1)
          self.mlp1 = nn.Linear(channel, channel)
          
          self.norm2 = RMSNorm(channel)
          self.mlp2 = nn.Sequential(nn.Linear(channel, expansion*channel),
                                    nn.ReLU(True),
                                    nn.Linear(expansion*hannel, channel))

          #self.act = HardSigmoid(True)
          self.act = nn.Sigmoid()
          
      def forward(self, x):
  
          T = x.size(-2)
          state  = self.norm1(x).cumsum(dim=1)
          scaler = torch.arange(1,T+1).cumsum(dim=-1).view(1,T,1)
          state  /= scaler

          out = self.act(self.mlp1(state))*x

          out = self.mlp2(self.norm2(out)) + out
          return out
