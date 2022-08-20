CUDA_VISIBLE_DEVICES=1  python train_videodata.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri/0 \
    --expname train_video_curri_ori  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 3 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1
    # --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1