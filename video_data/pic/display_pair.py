import torch
dir = '../coffee/pairs.th'
pair = torch.load(dir)
print(pair)