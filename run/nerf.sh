CUDA_VISIBLE_DEVICES=1 python train_mvs_nerf_finetuning_pl.py  \
    --dataset_name blender --datadir ./data/nerf_synthetic/chair \
    --expname lego-ft  --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1