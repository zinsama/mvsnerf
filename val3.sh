CUDA_VISIBLE_DEVICES=2  python val_owndata_multiset.py  \
    --dataset_name multiset --datadir ../dataset/slight/ori --multiset_num  3\
    --savedir trainset_result  \
    --expname slight_nopre --with_rgb_loss  --batch_size 1024  \
    --num_epochs 2 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 \
    --model_ckpt ./runs_fine_tuning/slight_nopre/ckpts
