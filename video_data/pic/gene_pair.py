import torch
dir = '../coffee'
pair = {}
pair["train"]=[]
pair["val"]=[]
for i in range(0,18):
    if i >= 9 and i <= 13:
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
