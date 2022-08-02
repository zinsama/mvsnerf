import torch
dir = './move/pairs.th'
pair = {}
pair["move_train_0"]=[1,2,3]
pair["move_train_1"]=[61,62,63]
pair["move_train_2"]=[121,122,123]
pair["move_train_3"]=[181,182,183]
pair["move_train_4"]=[241,242,243]
pair["move_train_5"]=[301,302,303]
pair["move_train_6"]=[361,362,363]
pair["move_train_7"]=[421,422,423]
pair["move_train_8"]=[481,482,483]
pair["move_train_9"]=[541,542,543]
pair["move_test"]=[]
pair["move_val_0"]=[]
pair["move_val_1"]=[]
pair["move_val_2"]=[]
pair["move_val_3"]=[]
pair["move_val_4"]=[]
pair["move_val_5"]=[]
pair["move_val_6"]=[]
pair["move_val_7"]=[]
pair["move_val_8"]=[]
pair["move_val_9"]=[]
for i in range(0,60):
    if i%12==0:
        pair["move_test"].extend([i,i+60,i+120,i+180,i+240,i+300,i+360,i+420,i+480,i+540])
        pair["move_val_0"].append(i)
        pair["move_val_1"].append(i+60)
        pair["move_val_2"].append(i+120)
        pair["move_val_3"].append(i+180)
        pair["move_val_4"].append(i+240)
        pair["move_val_5"].append(i+300)
        pair["move_val_6"].append(i+360)
        pair["move_val_7"].append(i+420)
        pair["move_val_8"].append(i+480)
        pair["move_val_9"].append(i+540)
    else:
        pass
        # pair["move_train_0"].append(i)
        # pair["move_train_1"].append(i+60)
        # pair["move_train_2"].append(i+120)
        # pair["move_train_3"].append(i+180)
        # pair["move_train_4"].append(i+240)
        # pair["move_train_5"].append(i+300)
        # pair["move_train_6"].append(i+360)
        # pair["move_train_7"].append(i+420)
        # pair["move_train_8"].append(i+480)
        # pair["move_train_9"].append(i+540)
print(pair)
torch.save(pair,dir)