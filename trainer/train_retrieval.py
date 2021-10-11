import pytorch_lightning as pl
import torch
import wandb
import torch.utils.data
from dataset.scene import SceneHandler
from model import get_retrieval_networks
from util import retrieval
from pathlib import Path
from model.loss import NTXentLoss
from dataset.patched_scene_dataset import PatchedSceneDataset
from util.misc import get_iou_matrix
from PIL import Image
from util.retrieval import RetrievalInterface
from util.visualization import render_visualizations_to_image


class RetrievalTrainingModule(pl.LightningModule):

    def __init__(self, config):
        super(RetrievalTrainingModule, self).__init__()
        self.save_hyperparameters(config)
        self.metric_keys = ['total_loss', 'loss_contrastive']
        self.fenc_input, self.fenc_target = get_retrieval_networks(model_config=self.hparams['retrieval_model'])
        self.nt_xent_loss = NTXentLoss(self.hparams['retrieval_training']['temprature'], True)
        self.scene_handlers = {
            'train': SceneHandler('train', config),
            'val': SceneHandler('val', config),
        }
        self.current_learning_rate = self.hparams['retrieval_training']['lr']
        self.retrieval_handler = RetrievalInterface(config['query'], config['retrieval_model']['latent_dim'])
        self.dataset = lambda split: PatchedSceneDataset(split, config[f'dataset_{split.split("_")[0]}'], self.scene_handlers[split.split('_')[0]])
        self.train_dataset = self.dataset('train')
        self.code_noise_func = (lambda x: torch.empty_like(x).normal_(mean=0, std=self.hparams['retrieval_training']['code_noise'])) if (self.hparams['retrieval_training']['code_noise'] > 0) else (lambda x: torch.zeros_like(x))
        self.input_noise_func = (lambda x: torch.empty_like(x).normal_(mean=0, std=self.hparams['retrieval_training']['input_noise'] * self.hparams['dataset_train']['voxel_size_target'])) if (self.hparams['retrieval_training']['input_noise'] > 0) else (lambda x: torch.zeros_like(x))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.fenc_input.parameters()) + list(self.fenc_target.parameters()), lr=self.hparams['retrieval_training']['lr'], weight_decay=5e-5)
        scheduler = []
        if self.hparams['retrieval_training']['scheduler'] is not None:
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams['retrieval_training']['scheduler'], gamma=0.5)]
        return [optimizer], scheduler

    # noinspection PyMethodOverriding
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 1500 and self.hparams['retrieval_training']['scheduler'] is not None:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams['retrieval_training']['lr']
        # update params
        self.current_learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams['retrieval_training']['batch_size'], shuffle=True, num_workers=self.hparams['retrieval_training']['num_workers'], drop_last=True, pin_memory=True)

    def val_dataloader(self):
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams['retrieval_training']['batch_size'], shuffle=False, num_workers=self.hparams['retrieval_training']['num_workers'], drop_last=False, pin_memory=True)

    @staticmethod
    def reshape_features(feats):
        feature_dim = feats.shape[1]
        feats = feats.permute((0, 2, 3, 4, 1)).reshape((-1, feature_dim))
        feats_normed = torch.nn.functional.normalize(feats, dim=1)
        return feats_normed

    def forward(self, batch):
        features_in = self.fenc_input(batch['input'])
        features_tgt = self.fenc_target(batch['target'])
        return features_in, features_tgt

    def step(self, batch, train=False):
        if train:
            batch['target'] = batch['target'] + self.input_noise_func(batch['target'])
        features_in, features_tgt = self.forward(batch)
        features_in_reshaped, features_tgt_reshaped = self.reshape_features(features_in), self.reshape_features(features_tgt)
        if train:
            features_in_reshaped = features_in_reshaped + self.code_noise_func(features_in_reshaped)
            features_tgt_reshaped = features_tgt_reshaped + self.code_noise_func(features_tgt_reshaped)
        loss_contrastive = torch.zeros(1, dtype=torch.float32).to(features_in.device)
        if self.hparams['retrieval_training']['loss']['contrastive'] > 0:
            iou_matrix = None
            if self.hparams['retrieval_training']['iou_scaling']:
                iou_matrix = get_iou_matrix(self.train_dataset.denormalize_target(batch['target']) <= 0.75 * self.hparams['dataset_train']['voxel_size_target']).repeat(2, 2)
            loss_contrastive = self.nt_xent_loss(features_in_reshaped, features_tgt_reshaped, iou_matrix)
        total_loss = loss_contrastive * self.hparams['retrieval_training']['loss']['contrastive']
        return total_loss, loss_contrastive

    def training_step(self, batch, batch_idx):
        total_loss, loss_contrastive = self.step(batch, train=True)
        self.log("learning_rate", self.current_learning_rate, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/contrastive_loss", loss_contrastive, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        total_loss, loss_contrastive = self.step(batch)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/contrastive_loss", loss_contrastive, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': total_loss}

    def log_metrics(self, key, metrics):
        metric_keys = [f"{key}/{m}" for m in ["iou", "cd", "precision", "recall"]]
        metric_dict = {}
        for i, m in enumerate(metrics):
            metric_dict[metric_keys[i]] = m
        self.logger.log_metrics(metric_dict, self.global_step)

    def validation_epoch_end(self, outputs):
        output_dir = Path("runs") / self.hparams["experiment"] / "visualization" / f"epoch_{self.current_epoch:04d}"
        output_dir.mkdir(exist_ok=True, parents=True)
        ds_train = self.dataset('train')
        ds_val = self.dataset('val')
        ds_vis = self.dataset('val_vis')
        ds_train_eval = self.dataset('train_eval')
        retrieval.create_dictionary(self.fenc_target, self.hparams['dictionary'], self.hparams['retrieval_model']['latent_dim'], ds_train, output_dir)
        print('[Eval-Train]')
        train_eval_retrievals = self.retrieval_handler.create_mapping_and_retrieve_nearest_scenes_for_all(self.fenc_input, output_dir, ds_train_eval, ds_train_eval, 1, True)
        t_iou, t_cd, t_precision, t_recall = retrieval.get_metrics_for_retrieval(train_eval_retrievals, ds_train_eval)
        print(f"Train rough IoU: {t_iou:.3f} | CD: {t_cd:.3f} | P: {t_precision:.3f} | R: {t_recall:.3f}")
        self.log_metrics("train", [t_iou, t_cd, t_precision, t_recall])
        print('[Eval-Train-GT]')
        train_eval_retrievals = self.retrieval_handler.create_mapping_and_retrieve_nearest_scenes_for_all(self.fenc_input, output_dir, ds_train_eval, ds_train_eval, 1, False)
        t_iou, t_cd, t_precision, t_recall = retrieval.get_metrics_for_retrieval(train_eval_retrievals, ds_train_eval)
        print(f"Train-GT rough IoU: {t_iou:.3f} | CD: {t_cd:.3f} | P: {t_precision:.3f} | R: {t_recall:.3f}")
        self.log_metrics("traingt", [t_iou, t_cd, t_precision, t_recall])
        print('[Eval-Validation]')
        val_retrievals = self.retrieval_handler.create_mapping_and_retrieve_nearest_scenes_for_all(self.fenc_input, output_dir, ds_train_eval, ds_val, 1, False)
        v_iou, v_cd, v_precision, v_recall = retrieval.get_metrics_for_retrieval(val_retrievals, ds_val)
        print(f"Val rough IoU: {v_iou:.3f} | CD: {v_cd:.3f} | P: {v_precision:.3f} | R: {v_recall:.3f}")
        self.log_metrics("val", [v_iou, v_cd, v_precision, v_recall])
        vis_retrievals = val_retrievals[[ds_val.scenes.index(x) for x in ds_vis.scenes], :, :, :, :]
        combined_retrievals = ds_vis.combine_retrievals(vis_retrievals, 0)
        combined_inputs = ds_vis.combine_inputs()
        combined_targets = ds_vis.combine_targets()
        (output_dir / "visualization_val_vis").mkdir(exist_ok=True)
        for cr_scene in combined_retrievals:
            self.scene_handlers["val"].visualize_target_chunk(combined_targets[cr_scene], output_dir / "visualization_val_vis" / f"{cr_scene}_gt.obj")
            self.scene_handlers["val"].visualize_target_chunk(combined_retrievals[cr_scene], output_dir / "visualization_val_vis" / f"{cr_scene}_pred.obj")
            self.scene_handlers["val"].visualize_input_chunk(combined_inputs[cr_scene], output_dir / "visualization_val_vis" / f"{cr_scene}_input.obj")
        render_visualizations_to_image(output_dir / "visualization_val_vis", output_dir / "render_val_vis")
        renders = [x for x in (output_dir / "render_val_vis").iterdir() if x.name.endswith('.jpg')]
        for im in renders:
            self.logger.experiment.log({f"visualization/{im.name}": [wandb.Image(Image.open(im))]}, step=self.global_step)


if __name__ == '__main__':
    from util import arguments
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    args = arguments.parse_arguments()
    args['no_retrievals'] = True
    seed_everything(args['seed'])
    logger = WandbLogger(project='Repatch3D[Retrieval]['+args['dataset_train']['dataset_name']+']'+args['suffix'], name=args['experiment'], id=args['experiment'])

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / args['experiment']), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, every_n_epochs=args['save_epoch'])

    model = RetrievalTrainingModule(args)

    trainer = Trainer(gpus=[0], num_sanity_val_steps=args['sanity_steps'], max_epochs=args['max_epoch'], limit_val_batches=args['val_check_percent'], callbacks=[checkpoint_callback],
                      val_check_interval=min(args['val_check_interval'], 1.0), check_val_every_n_epoch=max(1, args['val_check_interval']), resume_from_checkpoint=args['resume'], logger=logger, benchmark=True)

    trainer.fit(model)
