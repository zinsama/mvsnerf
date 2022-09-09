import random
from opt import config_parser
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import torchvision.transforms as transforms


from data import dataset_dict

# models
from models import *
from renderer import *
from utils import *
from data.ray_utils import ray_marcher,ray_marcher_fine

import imageio

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, depth_pred, depth_gt, mask=None):
        if None == mask:
            mask = depth_gt > 0
        loss = self.loss(depth_pred[mask], depth_gt[mask]) * 2 ** (1 - 2)
        return loss

class MVSSystem(LightningModule):
    def __init__(self, args):
        super(MVSSystem, self).__init__()
        self.args = args
        self.args.feat_dim = 8+3*4
        self.args.dir_dim = 3
        self.idx = 0

        self.loss = SL1Loss()

        # Create nerf model
        self.render_kwargs_train, self.render_kwargs_test, start, self.grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
        filter_keys(self.render_kwargs_train)

        # Create mvs model
        self.MVSNet = self.render_kwargs_train['network_mvs']
        self.render_kwargs_train.pop('network_mvs')

        dataset = dataset_dict[self.args.dataset_name]
        self.train_dataset = []
        self.val_dataset = []
        for i in range(self.args.multiset_num):
            self.train_dataset.append(dataset(args, split='train', idx=i, scene_num=i//self.args.scene_num))
        for i in range(self.args.multiset_num):
            self.val_dataset.append(dataset(args, split='val', idx=i, scene_num=i//self.args.scene_num))
        self.init_volume()
        # for i in range(self.args.multiset_num):
        #     print(self.volume[i].parameters())
        #     self.grad_vars += list(self.volume[i].parameters())
        #TODO
        # save_dir = f'runs_fine_tuning/{self.args.expname}/ckpts/'
        # os.makedirs(save_dir, exist_ok=True)
        # path = f'{save_dir}/volume.tar'
        # ckpt = {}
        # for i in range(self.args.multiset_num):
        #     ckpt[f'volume_{i}'] = self.volume[i].state_dict()
        # torch.save(ckpt, path)

    def init_volume(self):
        self.imgs = []
        for i in range(self.args.multiset_num):
            img,self.proj_mats,self.near_far_source,self.pose_source = self.train_dataset[i].read_source_views(device=device)
            self.imgs.append(img)
        # if args.ckpt:
        #     ckpts = torch.load(args.ckpt)
        #     if 'volume' not in ckpts.keys():
        #         self.MVSNet.train()
        #         with torch.no_grad():
        #             for i in range(self.args.multiset_num):
        #                 tmp_volume_feature, _, _ = self.MVSNet(self.imgs[i], self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)
        #                 # print(tmp_volume_feature.shape)
        #                 # for j in range(8):
        #                 #     for k in range(16):
        #                 #         img1 = transforms.ToPILImage()(tmp_volume_feature[0][j][k])
        #                 #         img1.save(f"./feature/{i}_{j}_{k}.png")
        #                 volume_feature.append(tmp_volume_feature)
        #                 del tmp_volume_feature
        #     else:
        #         volume_feature = ckpts['volume']['feat_volume']
        #         print('load ckpt volume.')
        # else:
        #     self.MVSNet.train()
        #     with torch.no_grad():
        #         for i in range(self.args.multiset_num):
        #             tmp_volume_feature, _, _ = self.MVSNet(self.imgs[i], self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)
        #             volume_feature.append(tmp_volume_feature)

        for i in range(self.args.multiset_num):
            self.imgs[i] = self.unpreprocess(self.imgs[i])

        # project colors to a volume
        self.density_volume = None
        # if args.use_color_volume or args.use_density_volume:
        #     D,H,W = volume_feature[0].shape[-3:]
        #     intrinsic, c2w = self.pose_source['intrinsics'][0].clone(), self.pose_source['c2ws'][0]
        #     intrinsic[:2] /= 4
        #     vox_pts = get_ptsvolume(H-2*args.pad,W-2*args.pad,D, args.pad,  self.near_far_source, intrinsic, c2w)
        #     self.color_feature = []
        #     for i in range(self.args.multiset_num):
        #         self.color_feature.append(build_color_volume(vox_pts, self.pose_source, self.imgs[i], with_mask=True).view(D,H,W,-1).unsqueeze(0).permute(0, 4, 1, 2, 3)) # [N,D,H,W,C]
        #     if args.use_color_volume:
        #         for i in range(self.args.multiset_num):
        #             volume_feature[i] = torch.cat((volume_feature[i], self.color_feature[i]),dim=1) # [N,C,D,H,W]

        #     if args.use_density_volume:
        #         self.vox_pts = vox_pts
        #         self.density_volume = [_ for _ in range(self.args.multiset_num)]
        #     else:
        #         del vox_pts
        # self.volume = []
        # for i in range(self.args.multiset_num):
        #     self.volume.append(RefVolume(volume_feature[i]).to(device)) 
        # del volume_feature

    def update_density_volume(self,idx):
        with torch.no_grad():
            network_fn = self.render_kwargs_train['network_fn']
            network_query_fn = self.render_kwargs_train['network_query_fn']

            D,H,W = self.volume[idx].feat_volume.shape[-3:]
            features = torch.cat((self.volume[idx].feat_volume, self.color_feature[idx]), dim=1).permute(0,2,3,4,1).reshape(D*H,W,-1)
            self.density_volume[idx] = render_density(network_fn, self.vox_pts, features, network_query_fn).reshape(D,H,W)
        del features

    def decode_batch(self, batch):
        rays = batch['rays'].squeeze()  # (B, 8)
        rgbs = batch['rgbs'].squeeze()  # (B, 3)
        idx = batch['idx']
        return rays, rgbs, idx

    def unpreprocess(self, data, shape=(1,1,3,1,1)):
        # to unnormalize image for visualization
        device = data.device
        mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
        std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
        return (data - mean) / std

    def forward(self):
        return


    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.grad_vars, lr=self.args.lrate, betas=(0.9, 0.999))
        scheduler = get_scheduler(self.args, self.optimizer)
        return [self.optimizer], [scheduler]

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
    

    # def train_dataloader(self):
    #     class myloader:
    #         def __init__(self, loaderlist, len):
    #             self.loaderlist = loaderlist
    #             self.len = len
    #         def __iter__(self):
    #             return self
    #         def __next__(self):
    #             return next(self.loaderlist[random.randint(0,self.len-1)])
    #     loader = []
    #     for dataset in self.train_dataset:
    #         loader.append(iter(DataLoader(dataset,
    #                       shuffle=True,
    #                       num_workers=8,
    #                       batch_size=args.batch_size,
    #                       pin_memory=True,
    #                       )))
    #     return myloader(loader,self.args.multiset_num)

    def train_dataloader(self):
        class myloader:
            def __init__(self, loaderlist, len, train_dataset):
                self.loaderlist = loaderlist
                self.len = len
                self.cur = 0
                self.train_dataset = train_dataset
            def __iter__(self):
                return self
            def __next__(self):
                self.cur = (self.cur+1)%self.len
                try:
                    return next(self.loaderlist[self.cur])
                except StopIteration:
                    self.reset()
                    raise StopIteration
            def reset(self):
                for i in range(self.len):
                    self.loaderlist[i] = iter(DataLoader(self.train_dataset[i],
                          shuffle=True,
                          num_workers=8,
                          batch_size=args.batch_size,
                          pin_memory=True
                          ))
                                        
        loader = []
        for dataset in self.train_dataset:
            loader.append(iter(DataLoader(dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=args.batch_size,
                          pin_memory=True
                          )))
        return myloader(loader,self.args.multiset_num,self.train_dataset)

    def val_dataloader(self):
        class myloader:
            def __init__(self, loaderlist, len, val_dataset):
                self.loaderlist = loaderlist
                self.len = len
                self.cur = 0
                self.val_dataset = val_dataset
            def __iter__(self):
                return self
            def __next__(self):
                self.cur = (self.cur+1)%self.len
                try:
                    return next(self.loaderlist[self.cur])
                except StopIteration:
                    self.reset()
                    raise StopIteration
            def reset(self):
                for i in range(self.len):
                    self.loaderlist[i] = iter(DataLoader(self.val_dataset[i],
                          shuffle=False,
                          num_workers=1,
                          batch_size=1,
                          pin_memory=True
                          ))
                                        
        loader = []
        for dataset in self.val_dataset:
            loader.append(iter(DataLoader(dataset,
                          shuffle=False,
                          num_workers=1,
                          batch_size=1,
                          pin_memory=True
                          )))
        return myloader(loader,self.args.multiset_num,self.val_dataset)

    def update_volume(self, idx):
        tmp_volume_feature, _, _ = self.MVSNet(self.imgs[idx], self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)        
        # del self.volume[idx]
        self.volume[idx] = RefVolume(tmp_volume_feature).to(device) 
        del tmp_volume_feature

    def training_step(self, batch, batch_nb):
        torch.cuda.empty_cache()
        rays, rgbs_target,idx = self.decode_batch(batch)
        idx = idx[0]
        # self.update_volume(idx) 
        volume_feature, _, _ = self.MVSNet(self.imgs[idx], self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)
        if args.use_density_volume and 0 == self.global_step%200:
            self.update_density_volume(idx)

        xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays, N_samples=args.N_samples,
                        lindisp=args.use_disp, perturb=args.perturb)

        # Converting world coordinate to ndc coordinate
        H,W = self.imgs[0].shape[-2:] 
        inv_scale = torch.tensor([W - 1, H - 1]).to(device)
        w2c_ref, intrinsic_ref = self.pose_source['w2cs'][0], self.pose_source['intrinsics'][0]
        xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale, near=self.near_far_source[0],far=self.near_far_source[1], pad=args.pad, lindisp=args.use_disp)

        # important sampleing
        if self.density_volume is not None and args.N_importance > 0:
            xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher_fine(rays, self.density_volume[idx], z_vals, xyz_NDC,
                                                                          N_importance=args.N_importance)
            xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                         near=self.near_far_source[0], far=self.near_far_source[1], pad=args.pad, lindisp=args.use_disp)

        # rendering
        # rgbs, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled, xyz_NDC, z_vals, rays_o, rays_d,
        #                                                self.volume[idx], self.imgs[idx],  **self.render_kwargs_train)
        rgbs, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled, xyz_NDC, z_vals, rays_o, rays_d,
                                                       volume_feature, self.imgs[idx],  **self.render_kwargs_train)

        log, loss = {}, 0
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], rgbs_target)
            # print(extras['rgb0'], rgbs_target)
            loss = loss + img_loss0
            # print(loss)
            psnr0 = mse2psnr2(img_loss0.item())
            self.log('train/PSNR0', psnr0.item(), prog_bar=True)


        ##################  rendering #####################
        if self.args.with_rgb_loss:
            img_loss = img2mse(rgbs, rgbs_target)
            loss += img_loss
            psnr = mse2psnr2(img_loss.item())

            with torch.no_grad():
                self.log('train/loss', loss, prog_bar=True)
                self.log('train/img_mse_loss', img_loss.item(), prog_bar=False)
                self.log('train/PSNR', psnr.item(), prog_bar=True)

        if self.global_step%4000 == 0:
            self.save_ckpt(f'{self.global_step}')

        return  {'loss':loss}

    def on_validation_epoch_start(self):
        self.volume = []
        for i in range(self.args.multiset_num):
            volume_feature, _, _ = self.MVSNet(self.imgs[i], self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)
            self.volume.append(volume_feature)
            del volume_feature

    def validation_step(self, batch, batch_nb):
        self.MVSNet.train()
        rays, img, idx = self.decode_batch(batch)
        idx = idx[0]
        
        # print(rays.shape,idx)
        img = img.cpu()  # (H, W, 3)
        # mask = batch['mask'][0]

        N_rays_all = rays.shape[0]

        ##################  rendering #####################
        keys = ['val_psnr_all']
        log = init_log({}, keys)
        with torch.no_grad():

            rgbs, depth_preds = [],[]
            for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                    N_samples=args.N_samples, lindisp=args.use_disp)

                # Converting world coordinate to ndc coordinate
                H, W = img.shape[:2]
                inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                w2c_ref, intrinsic_ref = self.pose_source['w2cs'][0], self.pose_source['intrinsics'][0].clone()
                intrinsic_ref[:2] *= args.imgScale_test/args.imgScale_train
                # volume_feature, _, _ = self.MVSNet(self.imgs[idx], self.proj_mats, self.near_far_source, pad=args.pad, lindisp=args.use_disp)
                xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                             near=self.near_far_source[0], far=self.near_far_source[1], pad=args.pad*args.imgScale_test, lindisp=args.use_disp)

                # important sampleing
                if self.density_volume is not None and args.N_importance > 0:
                    xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher_fine(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                    self.density_volume[idx], z_vals,xyz_NDC,N_importance=args.N_importance)
                    xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                    near=self.near_far_source[0], far=self.near_far_source[1],pad=args.pad, lindisp=args.use_disp)


                # rendering
                rgb, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled,
                                                                       xyz_NDC, z_vals, rays_o, rays_d,
                                                                       self.volume[idx], self.imgs[idx],
                                                                       **self.render_kwargs_train)
                # rgb, disp, acc, depth_pred, alpha, extras = rendering(args, self.pose_source, xyz_coarse_sampled,
                #                                                        xyz_NDC, z_vals, rays_o, rays_d,
                #                                                        self.volume[idx], self.imgs[idx],
                #                                                        **self.render_kwargs_train)

                rgbs.append(rgb.cpu());depth_preds.append(depth_pred.cpu())


            rgbs, depth_r = torch.clamp(torch.cat(rgbs).reshape(H, W, 3),0,1), torch.cat(depth_preds).reshape(H, W)
            # print(torch.isnan(rgbs).int().sum())
            img_err_abs = (rgbs - img).abs()
            img_err_abs2 = img_err_abs ** 2
            # log['val_psnr_all'] = mse2psnr(torch.mean(img_err_abs2[~img_err_abs2.isnan()]))
            log['val_psnr_all'] = mse2psnr(torch.mean(img_err_abs2))
            depth_r, _ = visualize_depth(depth_r, self.near_far_source)
            self.logger.experiment.add_images('val/depth_gt_pred', depth_r[None], self.global_step)
            # print(self.global_step,self.args.vis_steps,"!!!!!!!!!")
            # if (self.global_step+1)%self.args.vis_steps==0:
            img_vis = torch.stack((img, rgbs, img_err_abs.cpu()*5)).permute(0,3,1,2)
            self.logger.experiment.add_images('val/rgb_pred_err', img_vis, self.global_step)
            os.makedirs(f'runs_fine_tuning/{self.args.expname}/{self.args.expname}/',exist_ok=True)

            img_vis = torch.cat((img,rgbs,img_err_abs*10,depth_r.permute(1,2,0)),dim=1).numpy()
            imageio.imwrite(f'runs_fine_tuning/{self.args.expname}/{self.args.expname}/{self.global_step:08d}_{self.idx:02d}.png', (img_vis*255).astype('uint8'))
            self.idx += 1

        return log

    def validation_epoch_end(self, outputs):
        # print(outputs,"!!!!!")
        # print([x['val_psnr_all'] for x in outputs],"??????")
        if self.args.with_depth:
            mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
            mask_sum = torch.stack([x['mask_sum'] for x in outputs]).sum()
            # mean_d_loss_l = torch.stack([x['val_depth_loss_l'] for x in outputs]).mean()
            mean_d_loss_r = torch.stack([x['val_depth_loss_r'] for x in outputs]).mean()
            mean_abs_err = torch.stack([x['val_abs_err'] for x in outputs]).sum() / mask_sum
            mean_acc_1mm = torch.stack([x[f'val_acc_{self.eval_metric[0]}mm'] for x in outputs]).sum() / mask_sum
            mean_acc_2mm = torch.stack([x[f'val_acc_{self.eval_metric[1]}mm'] for x in outputs]).sum() / mask_sum
            mean_acc_4mm = torch.stack([x[f'val_acc_{self.eval_metric[2]}mm'] for x in outputs]).sum() / mask_sum

            self.log('val/d_loss_r', mean_d_loss_r, prog_bar=False)
            self.log('val/PSNR', mean_psnr, prog_bar=False)

            self.log('val/abs_err', mean_abs_err, prog_bar=False)
            self.log(f'val/acc_{self.eval_metric[0]}mm', mean_acc_1mm, prog_bar=False)
            self.log(f'val/acc_{self.eval_metric[1]}mm', mean_acc_2mm, prog_bar=False)
            self.log(f'val/acc_{self.eval_metric[2]}mm', mean_acc_4mm, prog_bar=False)
        # print(torch.stack([x['val_psnr_all'] for x in outputs]))
        mean_psnr_all = torch.stack([x['val_psnr_all'] for x in outputs]).mean()
        self.log('val/PSNR_all', mean_psnr_all, prog_bar=True)
        return


    def save_ckpt(self, name='latest'):
        save_dir = f'runs_fine_tuning/{self.args.expname}/ckpts/'
        os.makedirs(save_dir, exist_ok=True)
        path = f'{save_dir}/{name}.tar'
        ckpt = {
            'global_step': self.global_step,
            'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
            # 'volume': self.volume[0].state_dict(),
            'network_mvs_state_dict': self.MVSNet.state_dict()}
        if self.render_kwargs_train['network_fine'] is not None:
            ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
        torch.save(ckpt, path)
        # path2 = f'{save_dir}/volume.tar'
        # ckpt2 = {}
        # for i in range(self.args.multiset_num):
        #     ckpt2[f'volume_{i}'] = self.volume[i].state_dict()
        # torch.save(ckpt2, path2)
        print('Saved checkpoints at', path)

    def load_ckpt(self,ckpt_dir, name='latest' ):
        save_dir = ckpt_dir
        path = f'{save_dir}/{name}.tar'
        #ckpt = {
        #    'global_step': self.global_step,
        #    'network_fn_state_dict': self.render_kwargs_train['network_fn'].state_dict(),
        #    'volume': self.volume.state_dict(),
        #    'network_mvs_state_dict': self.MVSNet.state_dict()}
        ckpt = torch.load(path)
        print(ckpt.keys())
        print(ckpt['global_step'])
        # self.global_step = ckpt['global_step']
        self.render_kwargs_train['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
        # self.volume.load_state_dict(ckpt['volume'])
        self.MVSNet.load_state_dict(ckpt['network_mvs_state_dict'])

        if self.render_kwargs_train['network_fine'] is not None:
            #ckpt['network_fine_state_dict'] = self.render_kwargs_train['network_fine'].state_dict()
            self.render_kwargs_train['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])
        #torch.save(ckpt, path)
        print('Loaded checkpoints at', path)

    def load_volume(self,ckpt_dir,name='volume'):
        path = f'{ckpt_dir}/{name}.tar'
        ckpt = torch.load(path)
        for i in range(self.args.multiset_num):
            self.volume[i].load_state_dict(ckpt[f'volume_{i}'])


    
if __name__ == '__main__':
    torch.set_default_dtype(torch.float32)
    args = config_parser()
    system = MVSSystem(args)
    checkpoint_callback = ModelCheckpoint(os.path.join(f'runs_fine_tuning/{args.expname}/ckpts/','{epoch:02d}'),
                                          monitor='val/PSNR',
                                          mode='max',
                                          save_top_k=0)

    if(args.model_ckpt):
        system.load_ckpt(args.model_ckpt,name='mvsnerf-v0')
    # system.load_volume(args.model_ckpt)

    logger = loggers.TestTubeLogger(
        save_dir="runs_fine_tuning",
        name=args.expname,
        debug=False,
        create_git_tag=False
    )

    args.num_gpus, args.use_amp = 1, False
    trainer = Trainer(min_epochs=args.num_epochs,
                    #   max_steps=200000,
                      checkpoint_callback=checkpoint_callback,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=args.num_gpus,
                    #   auto_select_gpus=True,
                      distributed_backend='ddp' if args.num_gpus > 1 else None,
                      num_sanity_val_steps=1, #if args.num_gpus > 1 else 5,
                      # check_val_every_n_epoch = max(system.args.num_epochs//system.args.N_vis,1),
                      val_check_interval=5000,
                      benchmark=True,
                      precision=16 if args.use_amp else 32,
                      amp_level='O2')

    trainer.fit(system)
    system.save_ckpt()
    torch.cuda.empty_cache()