CUDA_VISIBLE_DEVICES=5  python val_owndata.py  \
    --dataset_name MyDataset --datadir ../../data/nerf_llff_data/beef_move  \
    --expname video_val2 --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/video_train/ckpts
    # --ckpt ./runs_fine_tuning/all_noshuffle_batch/ckpts/latest.tar --N_vis 1 \
    
