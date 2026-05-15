import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import mlflow
import os
from pathlib import Path
import yaml
import argparse
from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from model import DualEncoder, ImageEncoder, StrokeSetEncoder
from sigreg import build_sigreg_loss
from data import StrokeImageDataset, custom_collate_fn


class SigRegLightningModule(pl.LightningModule):
    """PyTorch Lightning module for dual encoder training with SigReg."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Build model
        model_config = config['model']
        dim_emb = int(model_config['dim_emb'])
        
        # Image encoder
        vit_layers = model_config.get('vit_layers', 4)
        self.image_encoder = ImageEncoder(dim_emb, vit_layers)
        
        # Stroke encoder
        tf_dim_model = int(model_config['tf_dim_model'])
        tf_layers = model_config.get('tf_layers', 4)
        tf_heads = model_config.get('tf_heads', 8)
        tf_dropout = model_config.get('tf_dropout', 0.1)
        tf_dim_ff = int(model_config['tf_dim_ff'])
        self.stroke_encoder = StrokeSetEncoder(
            dim_emb, tf_dim_model, tf_layers, tf_heads, tf_dropout, tf_dim_ff
        )
        
        # Alignment + configurable SIGReg regularization
        self.sigreg_loss = build_sigreg_loss(config['loss'])
        self._val_image_embs = []
        self._val_stroke_embs = []

    def _log_mlflow_metric(self, name: str, value, step: int) -> None:
        if mlflow.active_run() is None:
            return
        if getattr(self.trainer, "sanity_checking", False):
            return
        # MLflow metric names cannot contain '@'
        safe_name = name.replace("@", "_at_")
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
            value = value.detach().item()
        mlflow.log_metric(safe_name, float(value), step=step)
        
    def forward(self, images, stroke_labels, stroke_controls):
        """Forward pass through both encoders."""
        image_embeddings = self.image_encoder(images)
        stroke_embeddings = self.stroke_encoder(stroke_labels, stroke_controls)
        return image_embeddings, stroke_embeddings
    
    def training_step(self, batch, batch_idx):
        images, stroke_labels, stroke_controls = batch
        
        # Forward pass
        image_emb, stroke_emb = self(images, stroke_labels, stroke_controls)
        
        # SigReg loss
        loss, stats = self.sigreg_loss(image_emb, stroke_emb)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_sim_loss', stats['sim_loss'], on_step=True, on_epoch=True)
        self.log('train_reg_loss', stats['reg_loss'], on_step=True, on_epoch=True)
        self.log('train_std_x_mean', stats['std_x_mean'], on_step=False, on_epoch=True)
        self.log('train_std_y_mean', stats['std_y_mean'], on_step=False, on_epoch=True)
        self._log_mlflow_metric('train_loss_step', loss, self.global_step)
        self._log_mlflow_metric('train_sim_loss_step', stats['sim_loss'], self.global_step)
        self._log_mlflow_metric('train_reg_loss_step', stats['reg_loss'], self.global_step)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, stroke_labels, stroke_controls = batch
        
        # Forward pass
        image_emb, stroke_emb = self(images, stroke_labels, stroke_controls)
        
        # SigReg loss
        loss, stats = self.sigreg_loss(image_emb, stroke_emb)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_sim_loss', stats['sim_loss'], on_step=False, on_epoch=True)
        self.log('val_reg_loss', stats['reg_loss'], on_step=False, on_epoch=True)
        self._val_image_embs.append(image_emb.detach().cpu())
        self._val_stroke_embs.append(stroke_emb.detach().cpu())
        self._log_mlflow_metric('val_loss_step', loss, self.global_step)
        self._log_mlflow_metric('val_sim_loss_step', stats['sim_loss'], self.global_step)
        self._log_mlflow_metric('val_reg_loss_step', stats['reg_loss'], self.global_step)
        
        return loss

    def on_validation_epoch_start(self):
        self._val_image_embs = []
        self._val_stroke_embs = []

    def on_train_epoch_end(self):
        metrics = self.trainer.callback_metrics
        for key in ('train_loss', 'train_sim_loss', 'train_reg_loss', 'train_std_x_mean', 'train_std_y_mean'):
            if key in metrics:
                self._log_mlflow_metric(key, metrics[key], self.current_epoch)

    def on_validation_epoch_end(self):
        if not self._val_image_embs or not self._val_stroke_embs:
            return

        image_emb = torch.cat(self._val_image_embs, dim=0)
        stroke_emb = torch.cat(self._val_stroke_embs, dim=0)
        n = image_emb.shape[0]
        if n == 0:
            return

        image_emb = torch.nn.functional.normalize(image_emb, dim=1)
        stroke_emb = torch.nn.functional.normalize(stroke_emb, dim=1)
        sim = image_emb @ stroke_emb.T  # (N, N)

        # Cross-modal retrieval: image->stroke and stroke->image
        target = torch.arange(n, dtype=torch.long)
        ranks_i2s = torch.argsort(sim, dim=1, descending=True)
        ranks_s2i = torch.argsort(sim.T, dim=1, descending=True)

        pos_rank_i2s = (ranks_i2s == target.unsqueeze(1)).float().argmax(dim=1)
        pos_rank_s2i = (ranks_s2i == target.unsqueeze(1)).float().argmax(dim=1)

        for k in (1, 5, 10):
            k_eff = min(k, n)
            r_i2s = (pos_rank_i2s < k_eff).float().mean()
            r_s2i = (pos_rank_s2i < k_eff).float().mean()
            self.log(f'val_i2s_recall@{k}', r_i2s, on_step=False, on_epoch=True, prog_bar=(k == 1))
            self.log(f'val_s2i_recall@{k}', r_s2i, on_step=False, on_epoch=True, prog_bar=(k == 1))
            self._log_mlflow_metric(f'val_i2s_recall@{k}', r_i2s, self.current_epoch)
            self._log_mlflow_metric(f'val_s2i_recall@{k}', r_s2i, self.current_epoch)

        # Pair discrimination: diagonal pairs are positive, others are negative
        sim_np = sim.numpy()
        labels = np.eye(n, dtype=np.int32).reshape(-1)
        scores = sim_np.reshape(-1)
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
        self.log('val_pair_roc_auc', roc_auc, on_step=False, on_epoch=True)
        self.log('val_pair_pr_auc', pr_auc, on_step=False, on_epoch=True)
        self._log_mlflow_metric('val_pair_roc_auc', roc_auc, self.current_epoch)
        self._log_mlflow_metric('val_pair_pr_auc', pr_auc, self.current_epoch)

        metrics = self.trainer.callback_metrics
        for key in ('val_loss', 'val_sim_loss', 'val_reg_loss'):
            if key in metrics:
                self._log_mlflow_metric(key, metrics[key], self.current_epoch)
    
    def configure_optimizers(self):
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam').lower()
        lr = optimizer_config.get('lr', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Learning rate scheduler (optional)
        scheduler_config = optimizer_config.get('scheduler', None)
        if scheduler_config:
            scheduler_type = scheduler_config.get('type', 'cosine').lower()
            if scheduler_type == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.config['training']['max_epochs']
                )
            elif scheduler_type == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get('step_size', 10),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            else:
                raise ValueError(f"Unknown scheduler: {scheduler_type}")
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        
        return optimizer


class SigRegDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for SigReg training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage: str = None):
        dataset_config = self.config['dataset']
        
        if stage == 'fit' or stage is None:
            full_dataset = StrokeImageDataset(dataset_config)
            n = len(full_dataset)
            if n < 2:
                raise ValueError("Dataset must contain at least 2 samples for train/val split.")

            training_config = self.config.get('training', {})
            val_ratio = float(training_config.get('val_ratio', 0.2))
            if not (0.0 < val_ratio < 1.0):
                raise ValueError(f"training.val_ratio must be in (0, 1), got {val_ratio}")

            n_val = max(1, int(round(n * val_ratio)))
            n_val = min(n - 1, n_val)
            n_train = n - n_val

            seed = int(training_config.get('seed', 42))
            generator = torch.Generator()
            generator.manual_seed(seed)
            perm = torch.randperm(n, generator=generator)

            train_indices = perm[:n_train].tolist()
            val_indices = perm[n_train:].tolist()
            self.train_dataset = Subset(full_dataset, train_indices)
            self.val_dataset = Subset(full_dataset, val_indices)
    
    def train_dataloader(self):
        batch_size = self.config['training'].get('batch_size', 32)
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self):
        batch_size = self.config['training'].get('batch_size', 32)
        
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description='Train SigReg model with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                        help='Path to configuration file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if mlflow.active_run() is not None:
        raise RuntimeError(
            "Unexpected active MLflow run detected before startup. "
            "Run this script only via `mlflow run . -e train`."
        )

    run_id = os.getenv("MLFLOW_RUN_ID")
    if not run_id:
        raise RuntimeError(
            "No MLflow project run context found. Run this script via `mlflow run . -e train`."
        )

    with mlflow.start_run(run_id=run_id):
        # Log all config parameters
        mlflow.log_params(flatten_dict(config))
        
        # Initialize module and datamodule
        model = SigRegLightningModule(config)
        datamodule = SigRegDataModule(config)
        
        # Setup trainer
        trainer_config = config.get('trainer', {})
        trainer = pl.Trainer(
            max_epochs=config['training']['max_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            logger=False,  # We use MLflow directly
            log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
            enable_progress_bar=trainer_config.get('enable_progress_bar', True),
            enable_model_summary=trainer_config.get('enable_model_summary', True),
            gradient_clip_val=trainer_config.get('gradient_clip_val', None),
            precision=trainer_config.get('precision', '32-true'),
        )
        
        # Train
        trainer.fit(model, datamodule)
        
        # Log model artifacts
        current_run = mlflow.active_run()
        run_id_for_path = current_run.info.run_id if current_run else "unknown_run"
        model_save_path = Path('models') / run_id_for_path
        model_save_path.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(model_save_path / 'model.ckpt'))
        mlflow.pytorch.log_state_dict(
            model.state_dict(),
            artifact_path='model_state_dict'
        )


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary for MLflow logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)


if __name__ == '__main__':
    main()
