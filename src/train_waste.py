#!/usr/bin/env python
from getpass import getpass
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models
import torchvision.transforms.v2 as tfs

from torchmetrics import Accuracy

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from lightning.pytorch import seed_everything

from dataclasses import dataclass, asdict, field
from clearml import Task

torch.set_float32_matmul_precision('medium')

@dataclass
class CFG:
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr_fc: float = 1e-3
    batch_size: int = 128
    num_workers: int = 2
    data_dir: str = 'data/DATASET'
    models_dir: str = 'models'
    unfreeze_epoch: int = 10
    lr_backbone: int = 1e-4
    
class WasteLitModel(LightningModule):
    def __init__(self, 
                 unfreeze_epoch=10, 
                 lr_fc=1e-3, 
                 lr_backbone=1e-4,
                 task=None):
        super().__init__()
        
        self.lr_fc = lr_fc
        self.lr_backbone = lr_backbone
        self.backbone_unfrozen = False  # чтобы не разморозить дважды
        self.unfreeze_epoch = unfreeze_epoch
        self.task = task

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        #self.model = torch.compile(model)
        for param in self.model.parameters():
          param.requires_grad = False
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2)
        for param in self.model.fc.parameters():
          param.requires_grad = True

        self.loss = nn.CrossEntropyLoss()
        self.accuracy_train = Accuracy(task='multiclass', num_classes=2)
        self.accuracy_val = Accuracy(task='multiclass', num_classes=2)
        
    def forward(self, x):
        return self.model(x)
    
    def basic_step(self, batch, batch_idx, stage):
        x, y = batch
        preds = self(x)
        loss = self.loss(preds, y)
        acc = (preds.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        
        if self.task is not None:
            if stage == 'train':
                self.accuracy_train.update(preds, y)
            elif stage == 'val':
                self.accuracy_val.update(preds, y)
        return loss

    def on_train_epoch_start(self):
        if (self.current_epoch == self.unfreeze_epoch) and not self.backbone_unfrozen:
          for param in self.model.parameters():
            param.requires_grad = True
          self.backbone_unfrozen = True
          
    def _acc_compute(self, stage):
        if stage == 'train':
            acc_fn = self.accuracy_train
        elif stage == 'val':
            acc_fn = self.accuracy_val
        
        self.task.get_logger().report_scalar(
            title='Accuracy', 
            series=stage, 
            value=acc_fn.compute())
        acc_fn.reset()
    
    def on_train_epoch_end(self):
        if self.task is not None:
            self._acc_compute('train')
    
    def on_validation_epoch_end(self):
        if self.task is not None:
            self._acc_compute('val')
    
    def training_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self.basic_step(batch, batch_idx, 'test')
    
    def configure_optimizers(self):
        if not self.backbone_unfrozen:
          return optim.Adam(self.parameters(), lr=self.lr_fc)
        else:
          return optim.Adam([
              {"params": self.model.fc.parameters(), "lr": self.lr_fc},
              {"params": [p for n, p in self.model.named_parameters() if "fc" not in n], "lr": self.lr_backbone}
          ])


def check_clearml_env():
    os.environ['CLEARML_WEB_HOST'] = 'https://app.clear.ml/'
    os.environ['CLEARML_API_HOST'] = 'https://api.clear.ml'
    os.environ['CLEARML_FILES_HOST'] ='https://files.clear.ml'
    if os.getenv('CLEARML_API_ACCESS_KEY') is None:
        os.environ['CLEARML_API_ACCESS_KEY'] = getpass(prompt="Введите API Access токен: ")
    if os.getenv('CLEARML_API_SECRET_KEY') is None:
        os.environ['CLEARML_API_SECRET_KEY'] = getpass(prompt="Введите API Secret токен: ")

def main(arg):
    cfg = CFG()
    cfg.unfreeze_epoch = arg.unfreeze_epoch
    cfg.lr_fc = arg.lr
    seed_everything(cfg.seed)
    
    task = None
    if arg.clearml:
        try:
            task = Task.init(project_name="Waste classifier", task_name="Pytorch lightning model")
        except:
            check_clearml_env()
            task = Task.init(project_name="Waste classifier", task_name="Pytorch lightning model")
        cfg_dict = asdict(cfg)
        task.connect(cfg_dict)
        
    train_transforms = tfs.Compose([tfs.RandomRotation(30),
                                      tfs.RandomResizedCrop(224),
                                      tfs.RandomHorizontalFlip(),
                                      tfs.ToTensor(),
                                      tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    test_transforms = tfs.Compose([tfs.Resize(255),
                                     tfs.CenterCrop(224),
                                     tfs.ToTensor(),
                                     tfs.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(arg.data_dir+'/TRAIN', transform=train_transforms)
    val_data = datasets.ImageFolder(arg.data_dir+'/VAL', transform=test_transforms)
    test_data = datasets.ImageFolder(arg.data_dir+'/TEST', transform=test_transforms)
    
    train_loader = DataLoader(train_data, shuffle=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
            
    checkpoint_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
        filename="waste-resnet18-{epoch:02d}-{val_acc:.4f}",
        dirpath=cfg.models_dir
    )
    early_stop_cb = EarlyStopping(monitor='val_acc', mode='max', patience=3)

    model = WasteLitModel(unfreeze_epoch=arg.unfreeze_epoch, lr_fc=arg.lr, lr_backbone=1e-4, task=task)
    
    try:
        if arg.fast_dev_run:
            trainer = Trainer(fast_dev_run=arg.fast_dev_run)
            trainer.fit(model, train_loader, val_loader)
            print("!!!Тестовый прогон завершился УСПЕШНО!!!")
            return
        
        trainer = Trainer(
        max_epochs=arg.epochs,
        precision=16,
        accelerator=cfg.device,
        callbacks=[
            checkpoint_cb,
            early_stop_cb,
            ]
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model, test_loader)
        
        # --- Загружаем лучшие веса ---
        best_ckpt_path = checkpoint_cb.best_model_path
        best_model = WasteLitModel.load_from_checkpoint(best_ckpt_path)
        torch.save(best_model.model.state_dict(), cfg.models_dir+"waste_resnet18_best.pth")
        print(f"Модель сохранена в {cfg.models_dir}/waste_resnet18_best.pth")
    except Exception as exp:
        print(f"!!!EXCEPTION: {exp}")
        print("!!!Прогон завершился с ошибкой!!!")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python Lightning script")
    parser.add_argument('--fast_dev_run', type=bool, default=False, help='Run a single batch for quick debugging')
    parser.add_argument('--clearml', type=bool, default=False, help='Use ClearML for logging')
    parser.add_argument('--epochs', type=int, default=10, help='Run number of epochs')
    parser.add_argument('--data_dir', type=str, default='data/DATASET', help='Data path')
    parser.add_argument('--weights_path', type=str, default='models/waste_resnet18_best.pth', help='Where to save Weights of model')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--unfreeze_epoch', type=int, default=10)
    arg = parser.parse_args()
    main(arg)