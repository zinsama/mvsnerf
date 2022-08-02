import torch
dir = './slight_ten/move/pairs.th'
pair = {}
# pair["move_train"]=[181,182,183,184,185,200,201,202,203,204,235,236,237,238,239]
# pair["move_test"]=[181,182,183,184,185,200,201,202,203,204,235,236,237,238,239]
# pair["move_val"]=[181,182,183,184,185,200,201,202,203,204,235,236,237,238,239]
pair["move_train_0"]=[541,561,581]
pair["move_test"]=[]
pair["move_val_0"]=[]

for i in range(540,600,3):
    pair["move_test"].extend([i])
    pair["move_val_0"].extend([i])
    # pair["move_train_0"].append(i)
print(pair)
torch.save(pair,dir)