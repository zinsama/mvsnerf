CUDA_VISIBLE_DEVICES=0 python train_owndata_multiset_same.py  \
    --dataset_name multiset2 --datadir ./video_data/beef --multiset_num 5 \
    --expname train_beef_nobias_num5  --with_rgb_loss  --batch_size 2048   \
    --chunk 1024 --netchunk 1024 \
    --net_type v2  --netdepth 6 \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1  \