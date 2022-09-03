CUDA_VISIBLE_DEVICES=2  python train_owndata_multiset_same.py  \
    --dataset_name multiset2 --datadir ./video_data/coffee --multiset_num 1 \
    --expname train_video_coffee_1t  --with_rgb_loss  --batch_size 1024   \
    --chunk 2048 --netchunk 2048 --lrate 1e-4\
    --num_epochs 3 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \