CUDA_VISIBLE_DEVICES=2  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_all \
    --expname train_video_beef_all  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1