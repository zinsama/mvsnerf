CUDA_VISIBLE_DEVICES=0  python train_owndata_multiset_same.py  \
    --dataset_name multiset2 --datadir ./video_data/capture_image --multiset_num 9 \
    --expname train_video_multiset_num9  --with_rgb_loss  --batch_size 1024   \
    --chunk 2048 --netchunk 2048 \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \