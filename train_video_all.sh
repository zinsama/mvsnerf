CUDA_VISIBLE_DEVICES=0  python train_owndata_multiset_same.py  \
    --dataset_name multiset2 --datadir ./video_data/coffee --multiset_num 1 \
    --expname test1  --with_rgb_loss  --batch_size 1024   \
    --chunk 2048 --netchunk 2048 --lrate 1e-3\
    --num_epochs 3 --imgScale_test 1.0  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \
    # --model_ckpt ./ckpts --white_bkgd\