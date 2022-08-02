import torch
dir = './move/pairs.th'
pair = {}
pair["ori_train_0"]=[]
pair["ori_train_1"]=[]
pair["ori_train_2"]=[]
pair["ori_train_3"]=[]
pair["ori_train_4"]=[]
pair["ori_train_5"]=[]
pair["ori_train_6"]=[]
pair["ori_train_7"]=[]
pair["ori_train_8"]=[]
pair["ori_train_9"]=[]
pair["ori_test"]=[]
pair["ori_val_0"]=[]
pair["ori_val_1"]=[]
pair["ori_val_2"]=[]
pair["ori_val_3"]=[]
pair["ori_val_4"]=[]
pair["ori_val_5"]=[]
pair["ori_val_6"]=[]
pair["ori_val_7"]=[]
pair["ori_val_8"]=[]
pair["ori_val_9"]=[]
for i in range(0,60):
    if i%12==0:
        pair["ori_test"].extend([i,i+60,i+120,i+180,i+240,i+300,i+360,i+420,i+480,i+540])
        pair["ori_val_0"].append(i)
        pair["ori_val_1"].append(i+60)
        pair["ori_val_2"].append(i+120)
        pair["ori_val_3"].append(i+180)
        pair["ori_val_4"].append(i+240)
        pair["ori_val_5"].append(i+300)
        pair["ori_val_6"].append(i+360)
        pair["ori_val_7"].append(i+420)
        pair["ori_val_8"].append(i+480)
        pair["ori_val_9"].append(i+540)
    else:
        pair["ori_train_0"].append(i)
        pair["ori_train_1"].append(i+60)
        pair["ori_train_2"].append(i+120)
        pair["ori_train_3"].append(i+180)
        pair["ori_train_4"].append(i+240)
        pair["ori_train_5"].append(i+300)
        pair["ori_train_6"].append(i+360)
        pair["ori_train_7"].append(i+420)
        pair["ori_train_8"].append(i+480)
        pair["ori_train_9"].append(i+540)
print(pair)
torch.save(pair,dir)