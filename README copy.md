# Training & Testing Steps:
## Train:
train.sh\
`
CUDA_VISIBLE_DEVICES=0  python train_owndata.py  \
    --dataset_name own --datadir ../kubric/output7/ori \
    --expname 7  --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1
`

customized parameters: 
1. --datadir
2. --expname

After training, save checkpoint in ***./run_fine_tuning/expname/ckpts***

## Test：
val.sh \
`
CUDA_VISIBLE_DEVICES=2  python val_owndata.py  \
    --dataset_name own --datadir ../kubric/output7/move \
    --expname 7test  --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/7/ckpts/
`

customized parameters:
1. --datadir
2. --expname
3. --model_ckpt

## Example result:
origin scene: \
<img src="example_data/ori/train/1.png#pic_center" width="50%" ></img> \
moved scene: \
<img src="example_data/move/train/1.png#pic_center" width="50%" ></img> \
test result: \
<img src="example_test_result/test/00000000_00.png#pic_center" width="100%" ></img> \
<img src="example_test_result/test/00000000_02.png#pic_center" width="100%" ></img> \
<img src="example_test_result/test/00000000_04.png#pic_center" width="100%" ></img> \

