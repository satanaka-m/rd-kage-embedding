from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from fontTools.misc.transform import Transform

from cid_table import get_cid_table
from rasterize import rasterize, CidFace


def torch_uniform(a, b, size=(1,)):
    return (b-a) * torch.rand(*size) + a


class StrokeImageDataset(Dataset):
    def __init__(self, dataset_config):
        super().__init__()
        self.dataset_config = dataset_config
        self.width, self.height = dataset_config['width'], dataset_config['height']
        self.font = None  # フォントは遅延初期化
        self.setup()

    def setup(self):
        # ターゲットCIDリストの取得
        self.target_cid_list = get_cid_table(self.dataset_config['cids'])

        # KAGEデータ読み込み
        kage_pkl = Path(self.dataset_config['kage_pkl'])
        with open(kage_pkl, "rb") as f:
            self.kage = pickle.load(f)

    def setup_font(self):
        # マルチプロセッシング対応のためここでフォント読み込み
        # （freetypeのオブジェクトがpickleできないことが原因で、__init__内では読み込めない）
        font_path = Path(self.dataset_config['font_dir']) / self.dataset_config['font_filename']
        if font_path.exists():
            self.font = CidFace(font_path.resolve().as_posix())
        else:
            raise FileNotFoundError(f"Font file not found: {font_path}")

    def __len__(self):
        return len(self.target_cid_list)

    def __getitem__(self, idx):
        # フォントの遅延初期化
        if self.font is None:
            self.setup_font()

        target_cid = self.target_cid_list[idx]

        # augmentation
        if "augmentation" in self.dataset_config:
            aug = self.dataset_config['augmentation']
            if aug['scale']['min'] == aug['scale']['max']:
                a_scale = aug['scale']['min']
            else:
                a_scale = torch_uniform(aug['scale']['min'], aug['scale']['max']).item()
            d = aug['displacement']
            dx = torch_uniform(-d, d).item()
            dy = torch_uniform(-d, d).item()
        else:
            a_scale = 1.0
            dx = dy = 0.0
        
        transform = Transform().translate(dx, dy).translate(self.width/2, self.height/2).scale(a_scale*self.dataset_config['scale']).translate(-self.width/2, -self.height/2)

        # 画像のラスタライズ
        target_img = torch.tensor(rasterize(self.font, target_cid, self.width, self.height, transform).astype(np.float32)[None, :, :]/255)

        # ストロークフィーチャーの取得
        stroke_labels, stroke_controls = self._get_kage_features(target_cid)

        return target_img, stroke_labels, stroke_controls

    def _get_kage_features(self, cid: int):
        kage_strokes = self.kage[f"aj1-{cid:05d}"]
        stroke_labels = torch.zeros((len(kage_strokes), 3), dtype=torch.int64)
        stroke_controls = torch.zeros((len(kage_strokes), 4, 2), dtype=torch.float32)
        for i, stroke in enumerate(kage_strokes):
            stroke_labels[i, 0] = stroke.get_stroketype_idx()
            stroke_labels[i, 1] = stroke.get_startpointtype_idx()
            stroke_labels[i, 2] = stroke.get_endpointtype_idx()
            stroke_controls[i, :, :] = torch.tensor(stroke.get_overcomplete_controls())
        # 始点・終点以外を相対座標にする
        stroke_controls[:, 1, :] -=  stroke_controls[:, 0, :]
        stroke_controls[:, 2, :] -=  stroke_controls[:, 3, :]
        return stroke_labels, stroke_controls

def custom_collate_fn(batch):
    # stroke数が可変なので、paddingを行う
    # paddingは-1で行うことに注意

    target_img, stroke_labels, stroke_controls = list(zip(*batch))
    
    target_img = torch.stack(target_img)  # (B, 1, H, W)
    stroke_labels = torch.nn.utils.rnn.pad_sequence(stroke_labels, batch_first=True, padding_value=-1)  # (B, max(Nstrokes_i), 3)
    stroke_controls = torch.nn.utils.rnn.pad_sequence(stroke_controls, batch_first=True, padding_value=-1)  # (B, max(Nstrokes_i), 4, 2)

    return target_img, stroke_labels, stroke_controls


class StrokeImageDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset_config=None, val_dataset_config=None, batch_size=1):
        super().__init__()
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config
        self.batch_size = batch_size
        self.collate_fn = custom_collate_fn

    def train_dataloader(self):
        dataset = StrokeImageDataset(self.train_dataset_config)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn, pin_memory=True)
        return dataloader 

    def val_dataloader(self):
        if self.val_dataset_config is None:
            return None
        dataset = StrokeImageDataset(self.val_dataset_config)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=0, collate_fn=self.collate_fn, pin_memory=True)
        return dataloader 
