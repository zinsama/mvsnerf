CUDA_VISIBLE_DEVICES=0  python train_video_different.py  \
    --dataset_name MyDataset2 --datadir ./video_data/pic/beef_ref \
    --expname train_video_multiref_all  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 3 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    # --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1