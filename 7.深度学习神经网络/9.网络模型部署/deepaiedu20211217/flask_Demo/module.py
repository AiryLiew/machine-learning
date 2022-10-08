import torch
from torch import nn

class TestModel(nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(784, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.sequential(x)

if __name__ == '__main__':
    input = torch.randn(1, 784)
    model = TestModel()

    model.eval()
    traced_script_module = torch.jit.trace(model, input)
    traced_script_module.save("model.pt")


