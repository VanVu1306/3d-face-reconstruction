"""
MVF-Net Training Script
Main training loop for supervised pretraining and self-supervised training
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path

# Import our modules
from model_definition import create_mvfnet
from loss_functions import create_supervised_loss, create_selfsupervised_loss

# ============================================================================
# Configuration
# ============================================================================

class TrainConfig:
    # Paths
    data_root = "./processed_data"
    checkpoint_dir = "./checkpoints"
    log_dir = "./logs"
    bfm_path = "BFM/BFM_model_front.mat"
    
    # Training settings
    batch_size = 12
    num_workers = 4
    num_epochs_pretrain = 10
    num_epochs_finetune = 10
    
    # Optimizer settings
    lr_pretrain = 1e-5
    lr_finetune = 1e-6
    weight_decay = 1e-4
    
    # Loss weights (from paper)
    lambda1 = 0.1   # landmark loss (pretrain)
    lambda2 = 10    # pose loss (pretrain)
    lambda3 = 1     # 3DMM params loss (pretrain)
    lambda4 = 1     # regularization loss (pretrain)
    lambda5 = 1     # landmark loss (finetune)
    lambda6 = 10    # photometric loss (finetune)
    lambda7 = 0.1   # alignment loss (finetune)
    
    # Image settings
    img_size = 224
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Checkpointing
    save_freq = 1  # Save every N epochs
    log_freq = 10  # Log every N batches

# ============================================================================
# Dataset Classes
# ============================================================================

class W300LPDataset(Dataset):
    """Dataset for supervised pretraining on 300W-LP"""
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Load triplets metadata
        with open(os.path.join(data_path, '300wlp_triplets.json'), 'r') as f:
            self.triplets = json.load(f)
        
        # Split dataset
        n_train = int(len(self.triplets) * 0.9)
        if split == 'train':
            self.triplets = self.triplets[:n_train]
        else:
            self.triplets = self.triplets[n_train:]
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # Load images
        img_A = self._load_image(triplet['left']['image_path'])
        img_B = self._load_image(triplet['frontal']['image_path'])
        img_C = self._load_image(triplet['right']['image_path'])
        
        # Load ground truth
        gt = {
            'shape_params': torch.tensor(triplet['frontal']['shape_params'], dtype=torch.float32),
            'exp_params': torch.tensor(triplet['frontal']['exp_params'], dtype=torch.float32),
            'pose_A': torch.tensor(triplet['left']['pose_params'], dtype=torch.float32),
            'pose_B': torch.tensor(triplet['frontal']['pose_params'], dtype=torch.float32),
            'pose_C': torch.tensor(triplet['right']['pose_params'], dtype=torch.float32),
            'landmarks_A': torch.tensor(triplet['left']['landmarks'], dtype=torch.float32),
            'landmarks_B': torch.tensor(triplet['frontal']['landmarks'], dtype=torch.float32),
            'landmarks_C': torch.tensor(triplet['right']['landmarks'], dtype=torch.float32)
        }
        
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            img_C = self.transform(img_C)
        
        return {
            'img_A': img_A,
            'img_B': img_B,
            'img_C': img_C,
            'gt': gt
        }
    
    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img


class MultiPIEDataset(Dataset):
    """Dataset for self-supervised training on Multi-PIE"""
    
    def __init__(self, data_path, split='train', transform=None):
        self.data_path = data_path
        self.transform = transform
        
        # Load triplets metadata
        with open(os.path.join(data_path, 'multipie_triplets.json'), 'r') as f:
            self.triplets = json.load(f)
        
        # Split dataset
        n_train = int(len(self.triplets) * 0.9)
        if split == 'train':
            self.triplets = self.triplets[:n_train]
        else:
            self.triplets = self.triplets[n_train:]
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # Load images
        img_A = self._load_image(triplet['left']['image_path'])
        img_B = self._load_image(triplet['frontal']['image_path'])
        img_C = self._load_image(triplet['right']['image_path'])
        
        # Load detected landmarks (for optional landmark loss)
        landmarks = {
            'A': torch.tensor(triplet['left']['landmarks'], dtype=torch.float32),
            'B': torch.tensor(triplet['frontal']['landmarks'], dtype=torch.float32),
            'C': torch.tensor(triplet['right']['landmarks'], dtype=torch.float32)
        }
        
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
            img_C = self.transform(img_C)
        
        return {
            'img_A': img_A,
            'img_B': img_B,
            'img_C': img_C,
            'landmarks': landmarks
        }
    
    def _load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

# ============================================================================
# Training Functions
# ============================================================================

class Trainer:
    """Main trainer class"""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Initialize model
        print("Creating model...")
        self.model = create_mvfnet(config.bfm_path, config.img_size)
        self.model = self.model.to(config.device)
        
        # Initialize tensorboard
        self.writer = SummaryWriter(config.log_dir)
        
        print(f"Model created. Using device: {config.device}")
    
    def pretrain(self):
        """Supervised pretraining on 300W-LP"""
        print("\n" + "=" * 80)
        print("Starting Supervised Pretraining on 300W-LP")
        print("=" * 80)
        
        # Create dataset
        train_dataset = W300LPDataset(
            os.path.join(self.config.data_root, '300wlp'),
            split='train'
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Create loss function
        criterion = create_supervised_loss(
            self.model.bfm_model,
            lambda1=self.config.lambda1,
            lambda2=self.config.lambda2,
            lambda3=self.config.lambda3,
            lambda4=self.config.lambda4
        ).to(self.config.device)
        
        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_pretrain,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs_pretrain):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs_pretrain}")
            
            self.model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                img_A = batch['img_A'].to(self.config.device)
                img_B = batch['img_B'].to(self.config.device)
                img_C = batch['img_C'].to(self.config.device)
                gt = {k: v.to(self.config.device) for k, v in batch['gt'].items()}
                
                # Forward pass
                pred = self.model(img_A, img_B, img_C)
                
                # Compute loss
                loss, loss_dict = criterion(pred, gt, self.model.renderer)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Logging
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                if batch_idx % self.config.log_freq == 0:
                    for key, val in loss_dict.items():
                        self.writer.add_scalar(f'pretrain/{key}', val, global_step)
                
                global_step += 1
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            self.writer.add_scalar('pretrain/epoch_loss', avg_loss, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(f'pretrain_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint('pretrain_final.pth')
        print("\nPretraining completed!")
    
    def finetune(self, pretrain_checkpoint=None):
        """Self-supervised finetuning on Multi-PIE"""
        print("\n" + "=" * 80)
        print("Starting Self-Supervised Training on Multi-PIE")
        print("=" * 80)
        
        # Load pretrained model if provided
        if pretrain_checkpoint:
            print(f"Loading pretrained model from {pretrain_checkpoint}")
            self.load_checkpoint(pretrain_checkpoint)
        
        # Create dataset
        train_dataset = MultiPIEDataset(
            os.path.join(self.config.data_root, 'multipie'),
            split='train'
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        # Load PWC-Net for optical flow (placeholder - need actual implementation)
        # flow_network = load_pwcnet()
        flow_network = None  # Placeholder
        
        # Create loss function
        criterion = create_selfsupervised_loss(
            self.model.bfm_model,
            flow_network,
            lambda5=self.config.lambda5,
            lambda6=self.config.lambda6,
            lambda7=self.config.lambda7
        ).to(self.config.device)
        
        # Create optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_finetune,
            weight_decay=self.config.weight_decay
        )
        
        # Training loop
        global_step = 0
        for epoch in range(self.config.num_epochs_finetune):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs_finetune}")
            
            self.model.train()
            epoch_losses = []
            
            pbar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}")
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                img_A = batch['img_A'].to(self.config.device)
                img_B = batch['img_B'].to(self.config.device)
                img_C = batch['img_C'].to(self.config.device)
                landmarks = {k: v.to(self.config.device) for k, v in batch['landmarks'].items()}
                
                images = {'A': img_A, 'B': img_B, 'C': img_C}
                
                # Forward pass
                pred = self.model(img_A, img_B, img_C)
                
                # Compute loss
                loss, loss_dict = criterion(
                    pred, images, self.model.renderer, 
                    landmarks=landmarks
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Logging
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
                if batch_idx % self.config.log_freq == 0:
                    for key, val in loss_dict.items():
                        self.writer.add_scalar(f'finetune/{key}', val, global_step)
                
                global_step += 1
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
            self.writer.add_scalar('finetune/epoch_loss', avg_loss, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_freq == 0:
                self.save_checkpoint(f'finetune_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint('finetune_final.pth')
        print("\nFinetuning completed!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded: {filepath}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MVF-Net')
    parser.add_argument('--mode', type=str, default='pretrain',
                        choices=['pretrain', 'finetune', 'both'],
                        help='Training mode')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint for finetuning')
    parser.add_argument('--batch_size', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--num_epochs_pretrain', type=int, default=10,
                        help='Number of pretraining epochs')
    parser.add_argument('--num_epochs_finetune', type=int, default=10,
                        help='Number of finetuning epochs')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainConfig()
    config.batch_size = args.batch_size
    config.num_epochs_pretrain = args.num_epochs_pretrain
    config.num_epochs_finetune = args.num_epochs_finetune
    
    # Create trainer
    trainer = Trainer(config)
    
    # Run training
    if args.mode == 'pretrain' or args.mode == 'both':
        trainer.pretrain()
    
    if args.mode == 'finetune' or args.mode == 'both':
        pretrain_ckpt = args.pretrain_checkpoint
        if args.mode == 'both' and pretrain_ckpt is None:
            pretrain_ckpt = os.path.join(config.checkpoint_dir, 'pretrain_final.pth')
        trainer.finetune(pretrain_ckpt)
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

if __name__ == "__main__":
    main()
