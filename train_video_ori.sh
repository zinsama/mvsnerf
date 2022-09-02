CUDA_VISIBLE_DEVICES=3  python train_owndata2.py  \
    --dataset_name MyDataset --datadir ./video_data/pic/beef_curri2/0 \
    --expname train_video_curri_ori2  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  --lrate 5e-4\
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 --model_ckpt ./ckpts \