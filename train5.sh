CUDA_VISIBLE_DEVICES=5  python train_owndata.py  \
    --dataset_name own --datadir ../other_data/Scar \
    --expname Scar  --with_rgb_loss  --batch_size 4096  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1
