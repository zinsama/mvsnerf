CUDA_VISIBLE_DEVICES=0  python val_owndata_multiset.py  \
    --dataset_name multiset --datadir ./dataset/move --multiset_num 8 \
    --expname val_norunningnormf --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/normact_norunningnorm2/ckpts \
    # --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
