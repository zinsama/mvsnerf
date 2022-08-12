import torch
dir = './pairs.th'
pair = {}
pair["train"]=[]
pair["val"]=[]
for i in range(0,5):
    pair["val"].append(i)
for i in range(5,100):
    pair["train"].append(i)
print(pair)
torch.save(pair,dir)