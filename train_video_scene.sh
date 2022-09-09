CUDA_VISIBLE_DEVICES=5  python train_owndata_multiscene.py  \
    --dataset_name multiset3 --datadir ./video_data/mixed2 --multiset_num 4 --scene_num 2 \
    --expname multiscene3  --with_rgb_loss  --batch_size 1024   \
    --chunk 2048 --netchunk 2048\
    --num_epochs 3 --imgScale_test 1.0  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \
    # --model_ckpt ./ckpts --white_bkgd\