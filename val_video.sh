CUDA_VISIBLE_DEVICES=5  python val_owndata.py  \
    --dataset_name MyDataset --datadir  ./video_data/capture_image/10  \
    --expname val_video_multi_num10 --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./runs_fine_tuning/train_video_multiset_5/ckpts/latest.tar --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/train_video_multiset_5/ckpts \
    # --ckpt ./runs_fine_tuning/all_noshuffle_batch/ckpts/latest.tar --N_vis 1 \
    
