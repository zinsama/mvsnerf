import torch
dir = './move/pairs.th'
pair = torch.load(dir)
print(pair)