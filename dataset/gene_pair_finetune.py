import torch
dir = './slight_ten/move/pairs.th'
pair = {}
# pair = torch.load(dir)
# print(pair)
pair["move_train"]=[i for i in range(60,120,10)]
pair["move_test"]=[i for i in range(61,120,6)]
pair["move_val"]=[i for i in range(61,120,6)]
print(pair)
torch.save(pair,dir)