from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.utils.data
from tabulate import tabulate
from tqdm import tqdm
import numpy as np

from dataset.scene import SceneHandler
from model import get_unet_backbone, get_decoder, get_retrieval_backbone, get_attention_block
from model.attention import Unfold3D, Fold3D
from model.loss import NTXentLoss, get_cosine_similarity
from dataset.patched_scene_dataset import PatchedSceneDataset

from util.metrics import IoU, Chamfer3D, Precision, Recall
from util.misc import rename_state_dict


class RefinementTrainingModule(pl.LightningModule):

    def __init__(self, config):
        super(RefinementTrainingModule, self).__init__()
        self.save_hyperparameters(config)
        self.K = config['K']
        self.unet_backbone = get_unet_backbone(config)
        self.decoder = get_decoder(config)
        self.retrieval_backbone = get_retrieval_backbone(config)
        self.patched_attention_block = get_attention_block(config)
        self.scene_handlers = {
            'train': SceneHandler('train', config),
            'val': SceneHandler('val', config),
        }
        self.unfold_shape = Unfold3D(16, 1)
        self.unfold_features = Unfold3D(8, self.retrieval_backbone.nf)
        self.fold_shape = Fold3D(4, 16, 1)
        self.fold_features = Fold3D(4, 8, self.retrieval_backbone.nf)
        self.load_networks_if_needed()
        self.loss_ntxent = NTXentLoss(self.hparams["attn_temprature"], True)
        dataset = lambda split: PatchedSceneDataset(split, config[f'dataset_{split.split("_")[0]}'], self.scene_handlers[split.split('_')[0]])
        phase_func_list = [(self.optimizer_unet, self.training_step_unet), (self.optimizer_retrieval, self.training_step_retrieval), (self.optimizer_attention, self.training_step_attention), (self.optimizer_full, self.training_step_full)]
        self.init_opt_func = phase_func_list[config['current_phase']][0]
        self.training_step = phase_func_list[config['current_phase']][1]
        self.train_dataset, self.val_dataset = dataset('train'), dataset('val')
        self.train_vis_dataset, self.val_vis_dataset, self.train_eval_dataset = dataset('train_vis'), dataset('val_vis'), dataset('train_eval')
        self.metrics_train_fuse = self.metrics_train_nn1 = self.metrics_val_fuse = self.metrics_val_nn1 = None
        for m in ['metrics_train_fuse', 'metrics_train_nn1', 'metrics_val_fuse', 'metrics_val_nn1']:
            setattr(self, m, torch.nn.ModuleList([IoU(compute_on_step=False), Chamfer3D(compute_on_step=False), Precision(compute_on_step=False), Recall(compute_on_step=False)]))

    def training_step_unet(self, batch, _batch_idx):
        self.set_networks_to_unet_phase()
        self.augment_batch_data(batch)
        pred_shape = self.forward_backbone(batch)
        total_loss, loss_l1, loss_normal = self.loss_shape(pred_shape, batch)
        self.reset_network_state_to_train()
        return {'loss': total_loss}

    def training_step_retrieval(self, batch, _batch_idx):
        self.set_networks_to_retrieval_phase()
        self.augment_batch_data(batch)
        pred_shape = self.forward_retrieval(batch)
        total_loss, loss_l1, loss_normal = self.loss_shape(pred_shape, batch)
        self.reset_network_state_to_train()
        return {'loss': total_loss}

    def training_step_attention(self, batch, _batch_idx):
        self.set_networks_to_attention_phase()
        self.augment_batch_data(batch)
        x_attn_fpred, x_attn_ftgt, occupancy_attn = self.forward_attention(batch)
        loss_contrastive = self.compute_sliced_attn_nt_xent_loss(batch['target'].shape[0] * 8, x_attn_fpred, x_attn_ftgt, occupancy_attn)
        self.reset_network_state_to_train()
        return {'loss': loss_contrastive}

    def training_step_full(self, batch, _batch_idx):
        self.augment_batch_data(batch)
        self.reset_network_state_to_train()
        pred_shape, pred_back, pred_retr, x_attn_fpred, x_attn_ftgt, occupancy_attn = self.forward_full(batch)
        total_loss_fuse, loss_l1_fuse, loss_normal_fuse = self.loss_shape(pred_shape, batch)
        total_loss_back, loss_l1_back, loss_normal_back = self.loss_shape(pred_back, batch)
        total_loss_retr, loss_l1_retr, loss_normal_retr = self.loss_shape(pred_retr, batch)
        self.log_shape_loss('train_fuse', '', total_loss_fuse, loss_l1_fuse, loss_normal_fuse, on_epoch=False, on_step=True)
        loss_contrastive = self.compute_sliced_attn_nt_xent_loss(pred_retr.shape[0] * 8, x_attn_fpred, x_attn_ftgt, occupancy_attn)
        self.log('train_fuse/attn_contrastive', loss_contrastive, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        total_loss = total_loss_fuse + loss_contrastive * self.hparams['loss_attn_contrastive'] + total_loss_retr * self.hparams['loss_side_task_retr'] + total_loss_back * self.hparams['loss_side_task_unet']
        self.log('train_fuse/total_loss', total_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': total_loss}

    def training_step(self, batch, batch_idx):
        pass

    def forward_backbone(self, batch):
        x = self.unet_backbone(batch['input'])
        pred_shape = self.decoder(x)
        return pred_shape

    def forward_retrieval(self, batch):
        x = self.retrieval_backbone(self.unfold_shape(batch['target']))
        pred_shape = self.fold_shape(self.decoder(x))
        return pred_shape

    def forward_attention(self, batch):
        x_ = self.unet_backbone(batch['input'])
        x_target = self.fold_features(self.retrieval_backbone(self.unfold_shape(batch['target'])))
        pred_shape_ = self.decoder(x_)
        x_attn_fpred, x_attn_ftgt, occupancy_attn = self.patched_attention_block.get_features(x_, x_target, self.occupancy_from_prediction(self.network_pred_to_df(pred_shape_)))
        return x_attn_fpred, x_attn_ftgt, occupancy_attn

    def forward_full(self, batch):
        x_back = self.unet_backbone(batch['input'])
        retrievals = self.get_retrievals(batch['retrieval'])
        retrievals_plus_target = torch.cat([retrievals, batch['target']], dim=0)
        x_retrievals_plus_target = self.fold_features(self.retrieval_backbone(self.unfold_shape(retrievals_plus_target)))
        x_retrieval = x_retrievals_plus_target[:retrievals.shape[0], :, :, :, :]
        x_target = x_retrievals_plus_target[retrievals.shape[0]:, :, :, :, :]
        x = self.patched_attention_block(x_back, x_retrieval)
        pred_shape = self.decoder(x)
        pred_shape_retr = self.fold_shape(self.decoder(self.unfold_features(x_target)))
        pred_shape_back = self.decoder(x_back)
        x_attn_fpred, x_attn_ftgt, occupancy_attn = self.patched_attention_block.get_features(x_back, x_target, self.occupancy_from_prediction(self.network_pred_to_df(pred_shape_back)))
        return pred_shape, pred_shape_back, pred_shape_retr, x_attn_fpred, x_attn_ftgt, occupancy_attn

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.augment_batch_data(batch)
        pred_shape, pred_back, pred_retr, x_attn_fpred, x_attn_ftgt, occupancy_attn = self.forward_full(batch)
        loss_contrastive_attn = self.compute_sliced_attn_nt_xent_loss(pred_back.shape[0] * 8, x_attn_fpred, x_attn_ftgt, occupancy_attn).cpu().item()
        retrieval_1nn = batch['retrieval'][:, :1, :, :, :]
        split = ["val", "train"][dataloader_idx]
        suffix = ["", "_epoch"][dataloader_idx]
        metrics = [[self.metrics_train_fuse, self.metrics_train_nn1], [self.metrics_val_fuse, self.metrics_val_nn1]][dataloader_idx]
        self.log(f"{split}_full/attn_contrastive{suffix}", loss_contrastive_attn, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.get_evaluation_for_batch(f"{split}_full", suffix, metrics[0], self.network_pred_to_df(pred_shape), batch)
        self.get_evaluation_for_batch(f"{split}_nn1", suffix, metrics[1], self.val_dataset.denormalize_target(retrieval_1nn), batch)

    def validation_epoch_end(self, outputs):
        self.log("phase", self.hparams['current_phase'], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        table_data = [["split", "shape", "iou (rough)", "cd (rough)", "precision (rough)", "recall (rough)", "f1 (rough)"]]
        splittab = ["train", '', 'val', '']
        for idx, (split, pred_type, metrics) in enumerate([("train", "fuse", self.metrics_train_fuse), ("train", "nn1", self.metrics_train_nn1),
                                                           ("val", "fuse", self.metrics_val_fuse), ("val", "nn1", self.metrics_val_nn1)]):
            iou, cd, precision, recall = metrics[0].compute(), metrics[1].compute(), metrics[2].compute(), metrics[3].compute()
            f1 = 2 * (precision * recall) / (precision + recall)
            self.log(f"{split}_{pred_type}/iou", iou, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{split}_{pred_type}/cd", cd, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{split}_{pred_type}/precision", precision, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{split}_{pred_type}/recall", recall, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log(f"{split}_{pred_type}/f1", f1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            table_data.append([splittab[idx], pred_type, iou.item(), cd.item(), precision.item(), recall.item(), f1.item()])
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print('')
            print(tabulate(table_data, headers='firstrow', tablefmt='psql', floatfmt=".4f"))
            print('')
            visualization_datasets = [self.val_vis_dataset, self.train_vis_dataset]
            dataset_names = ['val', 'train']
            for ds_idx, ds in enumerate(visualization_datasets):
                if not (ds_idx == 1 and self.hparams['disable_train_vis']):
                    loader = torch.utils.data.DataLoader(ds, batch_size=self.hparams["batch_size"], shuffle=False, num_workers=self.hparams["num_workers"], drop_last=False)
                    nn1s, pred_shapes, pred_unets = [], [], []
                    for batch_idx, batch in enumerate(tqdm(loader, 'vis_infer')):
                        self.move_batch_to_cuda(batch, self.device)
                        self.augment_batch_data(batch)
                        pred_shape, pred_unet, _, _, _, _ = self.forward_full(batch)
                        nn1s.append(self.train_dataset.denormalize_target(batch['retrieval']).cpu().half())
                        pred_shapes.append(self.network_pred_to_df(pred_shape).cpu().half())
                        pred_unets.append(self.network_pred_to_df(pred_unet).cpu().half())
                    combined_pred_shapes = ds.combine_retrievals(torch.cat(pred_shapes, dim=0).numpy(), 0)
                    combined_inputs = ds.combine_inputs()
                    combined_targets = ds.combine_targets()
                    output_vis_path = Path("runs") / self.hparams['experiment'] / f"vis_{dataset_names[ds_idx]}" / f'{(self.global_step // 1000):05d}'
                    output_vis_path.mkdir(exist_ok=True, parents=True)
                    for cr_scene in tqdm(combined_targets, 'visualizing'):
                        self.scene_handlers["val"].visualize_target_chunk(combined_targets[cr_scene].astype(np.float32), output_vis_path / f"{cr_scene}_gt.obj")
                        self.scene_handlers["val"].visualize_target_chunk(combined_pred_shapes[cr_scene].astype(np.float32), output_vis_path / f"{cr_scene}_fuse.obj")
                        self.scene_handlers["val"].visualize_input_chunk(combined_inputs[cr_scene].astype(np.float32), output_vis_path / f"{cr_scene}_input.obj")

    def loss_shape(self, pred_shape, batch):
        loss_l1 = loss_normal = torch.zeros(1, dtype=torch.float32).to(pred_shape.device)
        if self.hparams['loss_reconstruction'] > 0:
            weights = self.adjust_weights(self.network_pred_to_df(pred_shape) >= self.scene_handlers['train'].target_trunc, batch)
            loss_l1 = (torch.abs(pred_shape - self.normalized_target_to_network_pred(batch['target'])) * weights).mean()
        if self.hparams['loss_normal'] > 0:
            loss_normal = (1 - get_cosine_similarity(self.train_dataset.compute_normals(self.network_pred_to_df(pred_shape)), batch['normals'])).mean()
        total_loss = self.hparams['loss_reconstruction'] * loss_l1 + self.hparams['loss_normal'] * loss_normal
        return total_loss, loss_l1, loss_normal

    def optimizer_unet(self):
        optimizer = torch.optim.Adam(list(self.unet_backbone.parameters()) + list(self.decoder.parameters()), lr=self.hparams["lr"])
        return [optimizer]

    def optimizer_retrieval(self):
        optimizer = torch.optim.Adam(list(self.retrieval_backbone.parameters()), lr=self.hparams["lr"])
        return [optimizer]

    def optimizer_attention(self):
        optimizer = torch.optim.Adam(list(self.patched_attention_block.parameters()), lr=self.hparams["lr"])
        return [optimizer]

    def optimizer_full(self):
        params = list(self.unet_backbone.parameters()) + list(self.decoder.parameters()) + list(self.retrieval_backbone.parameters()) + list(self.patched_attention_block.parameters())
        scheduler = []
        optimizer = torch.optim.Adam(params, lr=self.hparams["lr"])
        if self.hparams['scheduler'] is not None:
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams['scheduler'], gamma=0.5)]
        return [optimizer], scheduler

    def configure_optimizers(self):
        return self.init_opt_func()

    def compute_sliced_attn_nt_xent_loss(self, batch_size, x_attn_fpred, x_attn_ftgt, occupancy_attn):
        split_size = x_attn_fpred.shape[0] // batch_size
        total_unoccupied = 0
        loss_contrastive = torch.zeros(1, dtype=torch.float32).to(x_attn_fpred.device)
        # TODO: check attn occupancy, how much more than 1280
        self.log("train_full/attn_occupancy", (occupancy_attn > 0).sum().item(), on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        for b in range(batch_size):
            b_occ = occupancy_attn[b * split_size: (b + 1) * split_size] > 0
            if b_occ.sum() > 0 and total_unoccupied + b_occ.sum().detach().item() <= 1280:
                b_attn_fpred = x_attn_fpred[b * split_size: (b + 1) * split_size, :]
                b_attn_ftgt = x_attn_ftgt[b * split_size: (b + 1) * split_size, :]
                loss_contrastive = self.loss_ntxent(b_attn_fpred[b_occ], b_attn_ftgt[b_occ]).to(x_attn_fpred.device) + loss_contrastive
                total_unoccupied += b_occ.sum().detach().item()
        return loss_contrastive

    def get_evaluation_for_batch(self, key, suffix, metrics, pred_shape_df, batch):
        target_shape = self.train_dataset.denormalize_target(batch['target']) <= (self.scene_handlers['train'].target_voxel_size * 0.75)
        predicted_shape = pred_shape_df <= (self.scene_handlers['train'].target_voxel_size * 0.75)
        for metric in metrics:
            metric(predicted_shape, target_shape)
        total_loss, loss_l1, loss_normal = self.loss_shape(pred_shape_df, batch)
        self.log_shape_loss(key, suffix, total_loss, loss_l1, loss_normal, on_epoch=True, on_step=False)

    def augment_batch_data(self, batch):
        normal = self.train_dataset.compute_normals(self.train_dataset.denormalize_target(batch['target']))
        weights = torch.ones_like(batch['target']) * (1 + (batch['target'] < self.scene_handlers["train"].target_trunc).float() * (self.hparams["weight_occupied"] - 1)).float()
        empty = (batch['target'] >= self.scene_handlers['train'].target_trunc)
        batch['weights'] = weights
        batch['empty'] = empty
        batch['normals'] = normal

    def normalized_target_to_network_pred(self, target):
        return 2 * (self.train_dataset.denormalize_target(target) / self.scene_handlers['train'].target_trunc) - 1

    def network_pred_to_df(self, clamped_out):
        return (clamped_out + 1) * self.scene_handlers['train'].target_trunc / 2

    def occupancy_from_prediction(self, pred_shape_df):
        # TODO: test visually
        return torch.nn.functional.max_pool3d((pred_shape_df <= self.scene_handlers['train'].target_voxel_size * 0.75).float(), kernel_size=2, stride=2).bool().detach()

    @staticmethod
    def adjust_weights(pred_empty, batch):
        weights = batch['weights'].clone().detach()
        weights[batch['empty'] & pred_empty] = 0
        return weights

    def get_retrievals(self, retrievals):
        b, k, s = retrievals.shape[0:3]
        return retrievals[:, :self.K, :, :, :].reshape((b * self.K, 1, s, s, s))

    def log_shape_loss(self, stage, suffix, total, l1, normal, on_epoch, on_step):
        self.log(f'{stage}/shape{suffix}', total, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{stage}/l1{suffix}', l1, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{stage}/normal{suffix}', normal, on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True, sync_dist=True)

    def set_networks_to_unet_phase(self):
        self.unet_backbone.train()
        self.decoder.train()
        self.retrieval_backbone.eval()
        self.patched_attention_block.eval()

    def set_networks_to_retrieval_phase(self):
        self.unet_backbone.eval()
        self.decoder.eval()
        self.retrieval_backbone.train()
        self.patched_attention_block.eval()

    def set_networks_to_attention_phase(self):
        self.unet_backbone.eval()
        self.decoder.eval()
        self.retrieval_backbone.eval()
        self.patched_attention_block.train()

    def reset_network_state_to_train(self):
        self.unet_backbone.train()
        self.decoder.train()
        self.retrieval_backbone.train()
        self.patched_attention_block.train()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams["batch_size"], shuffle=True, num_workers=self.hparams["num_workers"], pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams["batch_size"], shuffle=False, num_workers=self.hparams["num_workers"], pin_memory=True, drop_last=False),
                torch.utils.data.DataLoader(self.train_eval_dataset, batch_size=self.hparams["batch_size"], shuffle=False, num_workers=self.hparams["num_workers"], pin_memory=True, drop_last=False)]

    def load_networks_if_needed(self):
        if self.hparams['resume'] is None:
            if self.hparams['unet_backbone_decoder_ckpt']:
                ckpt = torch.load(self.hparams['unet_backbone_decoder_ckpt'])
                self.unet_backbone.load_state_dict(rename_state_dict(ckpt['state_dict'], 'unet_backbone'))
                self.decoder.load_state_dict(rename_state_dict(ckpt['state_dict'], 'decoder'))
            if self.hparams['retrieval_backbone_ckpt']:
                ckpt = torch.load(self.hparams['retrieval_backbone_ckpt'])
                self.retrieval_backbone.load_state_dict(rename_state_dict(ckpt['state_dict'], 'retrieval_backbone'))
            if self.hparams['attention_block_ckpt']:
                ckpt = torch.load(self.hparams['attention_block_ckpt'])
                self.patched_attention_block.load_state_dict(rename_state_dict(ckpt['state_dict'], 'patched_attention_block'))

    @staticmethod
    def move_batch_to_cuda(batch, device):
        batch['input'] = batch['input'].to(device)
        batch['target'] = batch['target'].to(device)
        batch['retrieval'] = batch['retrieval'].to(device)

    def on_load_checkpoint(self, checkpoint):
        if type(self.init_opt_func()[0]) == list:
            checkpoint['optimizer_states'] = self.init_opt_func()[0][0].state
        else:
            checkpoint['optimizer_states'] = self.init_opt_func()[0].state


if __name__ == '__main__':
    from util import arguments
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from util.filesystem_logger import FilesystemLogger
    import os
    import gc

    args = arguments.parse_arguments()
    seed_everything(args['seed'])

    filesystem_logger = FilesystemLogger(args)
    logger = WandbLogger(project='Repatch3D[Refinement][' + args['dataset_train']['dataset_name'] + ']' + args['suffix'], name=args['experiment'], id=args['experiment'])

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / args['experiment']), save_top_k=-1, verbose=False, every_n_epochs=args['save_epoch'])

    max_loops = len(args['phase_change_epochs']) - args['current_phase']
    max_epochs = args['phase_change_epochs'] + [args['max_epoch']]
    for i in range(len(max_epochs) - 1):
        max_epochs[i + 1] = max_epochs[i] + max_epochs[i + 1]

    print('Max loops: ', max_loops)
    print('Max epochs: ', max_epochs)
    print('Starting phase', args['current_phase'])

    trainer = Trainer(gpus=-1, distributed_backend='ddp', num_sanity_val_steps=args['sanity_steps'], max_epochs=max_epochs[args['current_phase']], limit_val_batches=args['val_check_percent'], callbacks=[checkpoint_callback],
                      val_check_interval=min(args['val_check_interval'], 1.0), check_val_every_n_epoch=max(1, args['val_check_interval']), resume_from_checkpoint=args['resume'], logger=logger, benchmark=True)

    model = RefinementTrainingModule(args)
    trainer.fit(model)

    args['unet_backbone_decoder_ckpt'] = None
    args['retrieval_backbone_ckpt'] = None
    args['attention_block_ckpt'] = None

    for phase_idx in range(max_loops):
        del model
        gc.collect()
        args['current_phase'] += 1
        last_phase_ckpt = max([str(x) for x in (Path("runs") / args['experiment']).iterdir() if x.name.endswith('.ckpt')], key=os.path.getctime)
        print('Starting phase', args['current_phase'], '[' + last_phase_ckpt + ']', max_epochs[args['current_phase']])
        model = RefinementTrainingModule(args)
        trainer = Trainer(gpus=-1, accelerator='ddp', num_sanity_val_steps=0, max_epochs=max_epochs[args['current_phase']], limit_val_batches=args['val_check_percent'], callbacks=[checkpoint_callback],
                          val_check_interval=min(args['val_check_interval'], 1.0), check_val_every_n_epoch=max(1, args['val_check_interval']), logger=logger, benchmark=True, resume_from_checkpoint=last_phase_ckpt)
        trainer.fit(model)
