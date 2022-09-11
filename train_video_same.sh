CUDA_VISIBLE_DEVICES=4  python train_owndata_multiset_same.py  \
    --dataset_name multiset2 --datadir ./video_data/beef --multiset_num 5 \
    --expname train_hyper_beef_5  --with_rgb_loss  --batch_size 200   \
    --chunk 512 --netchunk 512 \
    --net_type v3  --netdepth 7 \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1  \
    # --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \