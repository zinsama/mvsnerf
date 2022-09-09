import torch
dir = '../beef'
valid_set = [3,7,11,15,19]
pair = {}
pair["train"]=[]
pair["val"]=[]
for i in range(0,20):
    if i in valid_set:
        pair["val"].append(i)
    else:
        pair["train"].append(i)
# for i in range(0,100):
#     if(i < 5):
#         pair["val"].append(i)
#     elif(i%20 >= 5):
#         pair["train"].append(i)

print(pair)
torch.save(pair,dir+f"/pairs.th")
