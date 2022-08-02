CUDA_VISIBLE_DEVICES=1  python train_single_volume.py  \
    --dataset_name multiset --datadir ../dataset/ori --multiset_num  8   \
    --expname sv_all_size8  --with_rgb_loss  --batch_size 1024   \
    --chunk 2048 --netchunk 2048 \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1  \