CUDA_VISIBLE_DEVICES=0 python train_single_volume.py  \
    --dataset_name multiset --datadir ../dataset/slight_ten/ori --multiset_num 9 \
    --expname sv_slight9  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 10 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1