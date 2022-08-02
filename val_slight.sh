CUDA_VISIBLE_DEVICES=3  python val_owndata_feature.py  \
    --dataset_name own --datadir ../dataset/slight_ten/move  \
    --expname vslight_val9 --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/slight9/ckpts  \
    # --ckpt ./runs_fine_tuning/slight_noshuffle_batch/ckpts/latest.tar --N_vis 1 \
    
    
