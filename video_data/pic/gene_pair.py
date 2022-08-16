import torch
dir = './pairs.th'
pair = {}
pair["train"]=[]
pair["val"]=[]
for i in range(0,20):
    # if(i%20 < 5):
    #     pair["val"].append(i)
    # else:
    #     pair["train"].append(i)
    pair["train"].append(i)
    pair["val"].append(i)
print(pair)
torch.save(pair,dir)