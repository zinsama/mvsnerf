import torch
tensor1 = torch.randn(10, 18, 4)
tensor1 = tensor1.unsqueeze(dim=-1)
print(tensor1.shape)
tensor2 = torch.randn(10, 18, 4, 6)
print(torch.matmul(tensor1, tensor2).size())