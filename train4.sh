CUDA_VISIBLE_DEVICES=3  python train_owndata.py  \
    --dataset_name test --datadir ../other_data/Easyship \
    --expname Easyship2  --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --N_samples 256 --N_importance 0  \
    --N_vis 1 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
    --use_color_volume --use_density_volume  \
    