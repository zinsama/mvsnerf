import torch
tensor1 = torch.randn(10, 18, 4)
tensor2 = torch.randn(4,6)
print(torch.matmul(tensor1, tensor2).size())