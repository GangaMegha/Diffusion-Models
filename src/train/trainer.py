import sys
sys.path.append("./Diffusion-Models/src/") 

from config import DATASET, CHECKPOINT_PATH, RESULT_PATH

import os 

import torch
from torchvision.utils import save_image

from train.helper_train import p_losses, num_to_groups
from diffusion.reverse_data_generate import sample
from diffusion.var_schedule import alpha_beta

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

    def train(self):

        min_val_loss = 1e8
        last_improv = -1

        train_loss_list, val_loss_list = [], []
        
        for epoch in range(self.cfg.get('epochs', 5)):
            train_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                batch_size = batch["pixel_values"].shape[0]
                batch = batch["pixel_values"].to(device)

                # Algorithm 1 line 3: sample t uniformally for every example in the batch
                t = torch.randint(0, self.cfg.get('T', 200), (batch_size,), device=device).long()

                loss = p_losses(self.model, batch, t, self.variance_dict, self.cfg.get('loss_type', "huber"), None)
                train_loss += loss.item()    

                if step % 100 == 0:
                    print("Loss:", loss.item())

                loss.backward()
                self.optimizer.step()

            # save generated images
            all_images_list = sample(self.model, self.variance_dict, self.cfg, sample_cnt=16)
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, os.path.join(RESULT_PATH, f'{self.dataset_name}/sample-{epoch}.png'), nrow=16)

            # End of epoch Validation loss
            val_loss = self.score(self.test_dataloader)
            print(f"\nEnd of Epoch {epoch} Train loss = {train_loss}, Val_loss = {val_loss}\n")
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)

            # Early stopping
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                last_improv = epoch
                self.save_checkpoint()
            if (epoch - last_improv) > self.cfg.get('patience', 5):
                break

        return train_loss_list, val_loss_list
    
    
    def score(self, dataloader):
        """Evaluates model performance on given data"""
        self.model.eval()
        loss = 0
        for batch in dataloader:
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, self.cfg.get('T', 200), (batch_size,), device=device).long()

            loss += p_losses(self.model, batch, t, self.variance_dict, self.cfg.get('loss_type', "huber"), None).item()

        self.model.train()

        return loss
    
    def save_checkpoint(self):
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, f'{self.dataset_name}/{self.model_name}.ckpt'))