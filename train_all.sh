CUDA_VISIBLE_DEVICES=4,5,6,7  \
python train_owndata.py  \
    --dataset_name own --datadir ../dataset/ori \
    --expname all2  --with_rgb_loss  --batch_size 4096  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1
