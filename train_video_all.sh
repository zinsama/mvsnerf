CUDA_VISIBLE_DEVICES=3  python train_owndata_multiset_same.py  \
    --dataset_name multiset2 --datadir ./video_data/spinach --multiset_num 10 \
    --expname train_video_spinach_10  --with_rgb_loss  --batch_size 1024   \
    --chunk 2048 --netchunk 2048\
    --num_epochs 1 --imgScale_test 1.0  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \
    # --model_ckpt ./ckpts --white_bkgd\
