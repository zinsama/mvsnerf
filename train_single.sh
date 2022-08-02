CUDA_VISIBLE_DEVICES=7 python train_owndata_multiset.py  \
    --dataset_name multiset --datadir ../dataset/single/ori --multiset_num  2\
    --expname single --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048 --use_color_volume --use_density_volume \
    --num_epochs 10 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1