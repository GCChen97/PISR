import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
from utils.misc import config_to_primitive

from systems.loss_pol import calc_aop_loss_terms, calc_aop_dop



@systems.register('pol-neus-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays

        dir_logs = self.save_dir.replace('save', 'logs')
        import os; os.makedirs(dir_logs, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=dir_logs)


    def forward(self, batch):
        return self.model(batch['rays'])

    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            if len(value) == 3 and not isinstance(value[0], list):
                value = [0] + value
            if isinstance(value[0], (int, float)):
                assert len(value) == 4
                start_step, start_value, end_value, end_step = value
                if isinstance(end_step, int):
                    current_step = self.global_step
                    value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
                elif isinstance(end_step, float):
                    current_step = self.current_epoch
                    value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            elif isinstance(value[0], list):
                current_step = self.global_step
                for value_i in value:
                    assert len(value_i) == 4
                    start_step, start_value, end_value, end_step = value_i
                    value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
                    if self.global_step < end_step:
                        break
            else:
                raise ValueError('Unknow config value: ', value)
        return value

    def preprocess_data(self, batch, stage):
        size_kernel = int(self.C(self.config.pol.train.size_kernel))
        step_kernel = int(self.C(self.config.pol.train.step_kernel))
        size_kernel = (size_kernel//2)*2 + 1
        len_kernel =  size_kernel*2-1
        # len_kernel =  size_kernel*4-1
        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            num_center_samples = self.train_num_rays//len_kernel
            self.writer.add_scalar('Int. Var./num_center_samples', num_center_samples, self.global_step)
            if self.config.model.batch_image_sampling:
                index = torch.randint(0, len(self.dataset.all_c2w), size=(num_center_samples,), device=self.dataset.all_c2w.device)
            else:
                index = torch.randint(0, len(self.dataset.all_c2w), size=(1,), device=self.dataset.all_c2w.device)
        if stage in ['train']:
            c2w = self.dataset.all_c2w[index]

            offset = 200
            x = torch.randint(
                offset, self.dataset.w*2-offset, size=(num_center_samples,), device=self.dataset.all_c2w.device
            )
            if size_kernel > 1:
                x = x.view(-1,1).repeat(1,len_kernel) # cross
                # add offsets
                ## cross pattern
                ### left
                x[:,1+(size_kernel//2)*0:1+(size_kernel//2)*1] \
                    -= step_kernel*(torch.arange(size_kernel//2)+1 ).view(1, size_kernel//2).to(self.rank)
                ### right
                x[:,1+(size_kernel//2)*1:1+(size_kernel//2)*2] \
                    += step_kernel*(torch.arange(size_kernel//2)+1 ).view(1, size_kernel//2).to(self.rank)

                x = x.view(-1,).clamp(offset, self.dataset.w*2-offset-1).long()
            
            y = torch.randint(
                0, self.dataset.h*2, size=(num_center_samples,), device=self.dataset.all_c2w.device
            )
            if size_kernel > 1:
                y = y.view(-1,1).repeat(1,len_kernel) # cross
                # add offsets
                ## cross pattern
                ### up
                y[:,1+(size_kernel//2)*2:1+(size_kernel//2)*3] \
                    -= step_kernel*(torch.arange(size_kernel//2)+1 ).view(1, size_kernel//2).to(self.rank)
                ### down
                y[:,1+(size_kernel//2)*3:1+(size_kernel//2)*4] \
                    += step_kernel*(torch.arange(size_kernel//2)+1 ).view(1, size_kernel//2).to(self.rank)

                y = y.view(-1,).clamp(0,2047).long()

            if self.config.model.batch_image_sampling:
                index = index.reshape([-1,1]).repeat(1, len_kernel).view(-1,)
                c2w = c2w[:,None,:,:].repeat(1,len_kernel,1,1).reshape([-1,3,4])

            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)

            rgb = self.dataset.all_images[index, y//2, x//2].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y//2, x//2].view(-1).to(self.rank)

            # pol
            if self.dataset.has_raw:
                aop_c, mask_oe, dop = calc_aop_dop(self.dataset.all_raws, index, y, x)

            else:
                aop_c = self.dataset.all_aops[index, y//2, x//2].view(-1,1)
                mask_oe = torch.zeros_like(aop_c)
                dop = self.dataset.all_dops[index, y//2, x//2].view(-1,1)

        else:
            c2w = self.dataset.all_c2w[index][0]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)

            rgb = None
            fg_mask = None

            # pol
            aop_c, mask_oe, dop = None, None, None

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])

        batch.update({
            'rays': rays,
            'rgb': rgb,
            'directions': directions,
            'c2w': c2w,
            'fg_mask': fg_mask,

            # pol
            'aop': aop_c,
            'mask_oe': mask_oe, # mask of overexposure points
            'dop': dop,

        })      

    def training_step(self, batch, batch_idx):
        if self.dataset.has_raw:
            size_kernel = int(self.C(self.config.pol.train.size_kernel))
        else:
            size_kernel = int(self.C(self.config.pol.train.size_kernel))
            size_kernel = 1 if size_kernel==0 else size_kernel
        size_kernel = (size_kernel//2)*2 + 1
        len_kernel =  size_kernel*2-1

        out = self(batch)
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / (1e-6+out['num_samples_full'].sum().item())))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
            # if self.global_step < 1500:
            #     self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
            #     self.train_num_rays = min(self.train_num_rays, 8192)
        
        # normal
        Rwc = batch['c2w'][:,:3,:3]
        Rwc[:,:3,1:3] *= -1.
        normals_w = out['comp_normal']

        if len(Rwc) > 1:
            normals = torch.bmm(normals_w.unsqueeze(1), Rwc).squeeze()#.float()
        elif len(Rwc) == 1:
            normals = torch.mm(normals_w, Rwc[0]).squeeze()#.float() # ! single-image sampling
        normals = F.normalize(normals, p=2, dim=-1)

        # rgb loss
        idx_rays_valid_full = out['rays_valid_full'][...,0]
        rgb_gt = batch['rgb'][idx_rays_valid_full]

        lambda_rgb_mse = self.C(self.config.system.loss.lambda_rgb_mse)
        if lambda_rgb_mse > 0:
            loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][idx_rays_valid_full], rgb_gt)
            self.log('train/loss_rgb_mse', loss_rgb_mse)
            self.writer.add_scalar('Loss/loss_rgb_mse', loss_rgb_mse, self.global_step)
            self.writer.add_scalar('Lambda/lambda_rgb_mse', lambda_rgb_mse, self.global_step)
            loss += lambda_rgb_mse * loss_rgb_mse

        lambda_rgb_l1 = self.C(self.config.system.loss.lambda_rgb_l1)
        if lambda_rgb_l1 > 0:
            loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][idx_rays_valid_full], rgb_gt)
            self.log('train/loss_rgb_l1', loss_rgb_l1)
            self.writer.add_scalar('Loss/loss_rgb_l1', loss_rgb_l1, self.global_step)
            self.writer.add_scalar('Lambda/lambda_rgb_l1', lambda_rgb_l1, self.global_step)
            loss += lambda_rgb_l1 * loss_rgb_l1

        # pol phase angle loss
        lambda_pol = self.C(self.config.pol.train.lambda_pol)
        loss_pa_ppa = self.config.pol.train.loss_pa_ppa
        loss_pa_type = self.config.pol.train.loss_pa_type
        loss_pa_normalize = self.config.pol.train.loss_pa_normalize
        if lambda_pol > 0:
            idx_rays_valid_full = out['rays_valid_full'][...,0]

            _dir = batch['directions'].float()[idx_rays_valid_full]
            _dir = F.normalize(_dir, p=2, dim=-1)
            _dir[:,1:3] *= -1.

            _aop = batch['aop'][idx_rays_valid_full, 0]
            _dop = batch['dop'][idx_rays_valid_full, 0].clip(0,1)

            _normals = normals[idx_rays_valid_full]

            nrx, nry = calc_aop_loss_terms(_aop, _normals, _dir, ppa=loss_pa_ppa, normalized=loss_pa_normalize)

            # diffuse/specular bias
            w_specular = _dop > 0.3
            bias_ = (1-w_specular.float())

            lx = (-torch.cos( torch.pi * nrx )*0.5+0.5) * 0.1
            ly = (-torch.cos( torch.pi * nry )*0.5+0.5) * 0.1
            aop_loss_raw = ly + bias_ * lx

            # weight by over exposure
            w_aop_loss = (1-batch['mask_oe'].float())[idx_rays_valid_full, 0]

            loss_aop = ( aop_loss_raw * w_aop_loss.view(-1) ).sum() / \
                (  w_aop_loss.sum() + 1e-6 )

            loss += lambda_pol * loss_aop

            self.writer.add_scalar('Lambda/lambda_aop', lambda_pol, self.global_step)
            self.writer.add_scalar('Loss/loss_aop', loss_aop, self.global_step)

        # smoothness
        lambda_smooth = self.C(self.config.pol.train.lambda_smooth)
        if lambda_smooth>0 and size_kernel > 1:
            idx_rays_valid = out['rays_valid'][...,0].view(-1, len_kernel)[:,0]

            _normals = out['comp_normal'].view(-1, len_kernel, 3)[idx_rays_valid]
            _normals_center = _normals[:, :1, :].detach()
            _normals_neighb = _normals[:, 1:, :]
            cos_normal = (_normals_center * _normals_neighb).sum(-1) # [-1, len_kernel]

            _weight = (1-batch['mask_oe'].float().view(-1, len_kernel)[:, 1:])[idx_rays_valid]

            loss_smoothness_raw = - cos_normal
            loss_smoothness = (
                    (_weight * loss_smoothness_raw).sum(dim=-1) / (_weight.sum(dim=-1) + 1e-6)
                ).mean()

            loss += lambda_smooth * loss_smoothness
            self.writer.add_scalar('Lambda/lambda_smooth', lambda_smooth, self.global_step)
            self.writer.add_scalar('Loss/loss_smoothness', loss_smoothness, self.global_step)

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        lambda_mask = self.C(self.config.system.loss.lambda_mask)
        has_mask = self.dataset.has_mask
        if lambda_mask > 0 and has_mask:
            loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float())
            self.log('train/loss_mask', loss_mask)
            self.writer.add_scalar('Loss/loss_mask', loss_mask, self.global_step)
            self.writer.add_scalar('Lambda/lambda_mask', lambda_mask, self.global_step)
            loss += loss_mask

        lambda_opaque = self.C(self.config.system.loss.lambda_opaque)
        if lambda_opaque > 0:
            loss_opaque = binary_cross_entropy(opacity, opacity)
            self.log('train/loss_opaque', loss_opaque)
            self.writer.add_scalar('Lambda/lambda_opaque', lambda_opaque, self.global_step)
            loss += lambda_opaque * loss_opaque

        lambda_sparsity = self.C(self.config.system.loss.lambda_sparsity)
        if lambda_sparsity > 0:
            loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
            self.log('train/loss_sparsity', loss_sparsity)
            self.writer.add_scalar('Lambda/lambda_sparsity', lambda_sparsity, self.global_step)
            loss += lambda_sparsity * loss_sparsity

        # for angelo
        # lambda_curvature = list_lambda_curvature[(milestones < self.global_step).sum()]
        lambda_curvature = self.C(self.config.system.loss.lambda_curvature)
        if lambda_curvature > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            loss += lambda_curvature * loss_curvature
            self.log('train/loss_curvature', loss_curvature)
            self.writer.add_scalar('Lambda/lambda_curvature', lambda_curvature, self.global_step)
            self.writer.add_scalar('Loss/loss_curvature', loss_curvature, self.global_step)

        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        lambda_distortion = self.C(self.config.system.loss.lambda_distortion)
        if lambda_distortion > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            self.writer.add_scalar('Lambda/lambda_distortion', lambda_distortion, self.global_step)
            loss += lambda_distortion * loss_distortion

        lambda_distortion_bg = self.C(self.config.system.loss.lambda_distortion_bg)
        if self.config.model.learned_background and lambda_distortion_bg > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            self.writer.add_scalar('Lambda/lambda_distortion_bg', lambda_distortion_bg, self.global_step)
            loss += lambda_distortion_bg * loss_distortion_bg

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        self.writer.add_scalar('Loss/loss_all', loss, self.global_step)
        self.writer.add_scalar('Loss/loss_eikonal', loss_eikonal, self.global_step)

        if (self.global_step) % self.config.pol.val.export_freq == 0 and self.global_step > 0:
            try:
                print("Exporting mesh...")
                mesh_pred = self.export()
            except IndexError:
                print("Exported mesh failed")

        return {
            'loss': loss
        }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)

        psnr = 0.
        W, H = self.dataset.img_wh
        W *= 2; H *= 2

        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            # {'type': 'rgb', 'img': batch['rgb'].reshape([H, W, 1]).repeat(1,1,3), 'kwargs': {'data_format': 'HWC'}},
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])

        del out # avoid OOM
        return {
            'psnr': psnr,
            'index': batch['index']
        }

    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)         

    def test_step(self, batch, batch_idx):
        # psnr = 0.

        # return {
        #     'psnr': psnr,
        #     'index': batch['index']
        # }    
        out = self(batch)

        # psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        psnr = 0.
        W, H = self.dataset.img_wh # [1224, 1024] due to downsample in dataloader

        vis_opacity = out['opacity'].view(H, W, 1)

        vis_rgb = out['comp_rgb'].view(H, W, 3)
        vis_rgb = vis_rgb + (1.0 - vis_opacity) * torch.ones_like(vis_rgb)

        vis_normal = out['comp_normal'].view(H, W, 3) * vis_opacity
        vis_normal = vis_normal + (1.0 - vis_opacity) * torch.ones_like(vis_normal)

        # sequence
        self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
            # {'type': 'rgb', 'img': batch['rgb'].view([H, W, 3]), 'kwargs': {'data_format': 'HWC'}},
        ] + ([
            # {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            # {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': vis_rgb, 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            # {'type': 'grayscale', 'img': out['depth'].view(H, W)*out['opacity'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': vis_normal, 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        del out
        return {
            'psnr': psnr,
            'index': batch['index']
        }      
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

            # sequence
            self.save_img_sequence(
                f"it{self.global_step}-test",
                f"it{self.global_step}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=30
            )
            
            if self.global_step == int(self.config.trainer.max_steps):
                # change range for export
                mc_range = self.config.export.export_mc_range
                if isinstance(mc_range, float):
                    self.model.geometry.mc_range = mc_range
                else:
                    self.model.geometry.mc_range = config_to_primitive(mc_range)

                # increase resolution for vis
                self.model.geometry.helper.verts = None
                resolution = 512
                self.model.geometry.helper.resolution = resolution
                print('\n[test_epoch_end] marching cube resolution', resolution, '\n')
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.model.geometry.helper.resolution}.obj",
            **mesh
        )
        return mesh
