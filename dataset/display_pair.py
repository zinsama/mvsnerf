import torch
dir = '../mvsnerf/configs/pairs.th'
pair = torch.load(dir)
print(pair)