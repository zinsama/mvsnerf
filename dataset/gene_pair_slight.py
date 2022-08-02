import torch
dir = './single/ori/pairs.th'
pair = {}
pair["ori_train_0"]=[]
pair["ori_train_1"]=[]
pair["ori_test"]=[]
pair["ori_val_0"]=[]
pair["ori_val_1"]=[]
for i in range(0,60):
    if i%6==0:
        pair["ori_test"].extend([i,i+60])
        pair["ori_val_0"].append(i)
        pair["ori_val_1"].append(i+60)
    else:
        pair["ori_train_0"].append(i)
        pair["ori_train_1"].append(i+60)
print(pair)
torch.save(pair,dir)