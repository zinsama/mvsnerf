CUDA_VISIBLE_DEVICES=1  python train_owndata_multiset.py  \
    --dataset_name multiset --datadir ./dataset/ori --multiset_num  8   \
    --expname normact_batchnorm2  --with_rgb_loss  --batch_size 1024   \
    --chunk 2048 --netchunk 2048 \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_vis 1  \
    --model_ckpt ./runs_fine_tuning/normact_batchnorm2/ckpts \
    # --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \