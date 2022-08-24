CUDA_VISIBLE_DEVICES=6  python val_owndata_dif.py  \
    --dataset_name MyDataset2 --datadir  ./video_data/pic/beef_ref \
    --expname val_curri_multiref_ori --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./runs_fine_tuning/train_video_multiref_ori/ckpts/latest.tar  --N_vis 1\
    --model_ckpt ./runs_fine_tuning/train_video_multiref_ori/ckpts
    # --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
    # --ckpt ./runs_fine_tuning/all_noshuffle_batch/ckpts/latest.tar --N_vis 1 \
    
