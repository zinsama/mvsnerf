CUDA_VISIBLE_DEVICES=7  python train_owndata.py  \
    --dataset_name own --datadir ../dataset/slight/ori \
    --expname slight_shuffle  --with_rgb_loss  --batch_size 2048  \
    --chunk 2048 --netchunk 2048  \
    --num_epochs 3 --imgScale_test 1.0 --white_bkgd  --pad 0 \
    --shuffle --N_vis 1