import torch
dir = './beef_all/pairs.th'
pair = torch.load(dir)
print(pair)