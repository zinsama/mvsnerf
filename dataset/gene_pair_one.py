import torch
dir = './single/ori/pairs.th'
pair = {}
pair["ori_train"]=[]
pair["ori_test"]=[]
pair["ori_val"]=[]
for i in range(0,60):
    if i%6==0:
        pair["ori_test"].extend([i])
        pair["ori_val"].extend([i])
    else:
        pair["ori_train"].append(i)
print(pair)
torch.save(pair,dir)