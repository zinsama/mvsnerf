CUDA_VISIBLE_DEVICES=0  python train_mvs_nerf_pl.py \
    --with_depth  --imgScale_test 1.0 \
    --expname mvs-nerf-is-all-your-need \
    --num_epochs 6 --N_samples 128 --use_viewdirs --batch_size 1 \
    --dataset_name blender \
    --datadir data/nerf_synthetic \
    --N_vis 1