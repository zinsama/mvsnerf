import torch
dir = './beef_ref/pairs.th'
pair = torch.load(dir)
print(pair)