CUDA_VISIBLE_DEVICES=0  python train_mvs_nerf_finetuning_pl.py  \
    --dataset_name own --datadir ../dataset/slight/move \
    --expname slight_noshuffle_batch_finetune --with_rgb_loss  --batch_size 1024  \
    --num_epochs 1 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./runs_fine_tuning/slight_noshuffle_batch/ckpts/latest.tar --N_vis 1 \
    # --ckpt ./ckpts/mvsnerf-v0.tar \
    # --model_ckpt ./runs_fine_tuning/slight_noshuffle_batch/ckpts
    # --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
    # --model_ckpt ./runs_fine_tuning/slight_noshuffle_batch/ckpts
