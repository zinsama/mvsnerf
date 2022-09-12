CUDA_VISIBLE_DEVICES=4 python train_owndata_multiset_same.py  \
    --dataset_name multiset2 --datadir ./video_data/beef --multiset_num 5 \
    --expname train_hyper2_beef_5  --with_rgb_loss  --batch_size 256   \
    --chunk 512 --netchunk 512 \
    --net_type v4  --netdepth 7 \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./runs_fine_tuning/train_hyper2_beef_5/ckpts/28000.tar --N_vis 1  \