import torch
dir = './beef_all/pairs.th'
pair = {}
pair["train"]=[]
pair["val"]=[]
for i in range(0,100):
    if(i < 5):
        pair["val"].append(i)
    elif(i%20 >= 5):
        pair["train"].append(i)
print(pair)
torch.save(pair,dir)