import sys
sys.path.append("./Diffusion-Models/src/") 

from config import DATASET, CHECKPOINT_PATH, RESULT_PATH

import os 

import torch
from torchvision.utils import save_image

from train.helper_train import p_losses
from diffusion.reverse_data_generate import sample
from diffusion.var_schedule import alpha_beta

from data_loader import reverse_transform

from train.metrics import FID, IS

device = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    """Class that handles training"""
    def __init__(self, model, train_cfg, model_name, dataset_name, train_dataloader, test_dataloader):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=train_cfg.get('lr', 1e-3), 
                                        weight_decay=train_cfg.get('weight_decay', 0.0))
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.cfg = train_cfg

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.variance_dict = alpha_beta(self.cfg.get('T', 200), schedule=self.cfg.get("var_schedule", "linear"))

        self.FID = FID()
        self.IS = IS()

    def train(self):

        min_val_loss = 1e8
        last_improv = -1
        n_epochs = self.cfg.get('epochs', 5)

        train_loss_list, val_loss_list, FID_list, IS_list = [], [], [], []
        
        for epoch in range(n_epochs):
            train_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.cfg.get('T', 200), (batch_size,), device=device).long()

                loss = p_losses(self.model, batch, t, self.variance_dict, self.cfg.get('loss_type', "huber"), None, self.cfg.get('clip', 1))
                train_loss += loss.item()    

                loss.backward()
                self.optimizer.step()

            # save generated images
            all_images_list = sample(self.model, self.variance_dict, self.cfg, sample_cnt=16)
            all_images = torch.cat(all_images_list, dim=0)
            save_image((all_images+1)*0.5, os.path.join(RESULT_PATH, f'{self.dataset_name}/sample-{epoch+1}.png'), nrow=16)

            # End of epoch Validation loss
            val_loss = self.score(self.test_dataloader)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            # Early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                last_improv = epoch
                self.save_checkpoint(epoch)

            # Calculate FID and InceptionScore
            if epoch%5==0 or epoch==(n_epochs-1) or (epoch - last_improv) > self.cfg.get('patience', 5):
                all_images_list = sample(self.model, self.variance_dict, self.cfg, sample_cnt=100)
                all_images = reverse_transform()(torch.cat(all_images_list, dim=0))
                FID_list.append(self.FID(all_images, reverse_transform()(batch.cpu()), self.cfg["grayscale"]))
                IS_list.append(self.IS(all_images, self.cfg["grayscale"]))
                print(f"\nEnd of Epoch {epoch+1} Train loss = {train_loss}, Val_loss = {val_loss}, FID = {FID_list[-1]}, IS = {IS_list[-1]} \n")
            else:
                FID_list.append(-1)
                IS_list.append(-1)
                print(f"\nEnd of Epoch {epoch+1} Train loss = {train_loss}, Val_loss = {val_loss} \n")
                
            if (epoch - last_improv) > self.cfg.get('patience', 5):
                break



        del self.FID
        del self.IS
        return [train_loss_list, val_loss_list, FID_list, IS_list]
    
    
    def score(self, dataloader):
        """Evaluates model performance on given data"""
        self.model.eval()
        loss = 0
        for batch in dataloader:
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, self.cfg.get('T', 200), (batch_size,), device=device).long()

            loss += p_losses(self.model, batch, t, self.variance_dict, self.cfg.get('loss_type', "huber"), None, self.cfg.get('clip', 1)).item()

        self.model.train()

        return loss
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, f'{self.dataset_name}/{self.model_name}.ckpt'))