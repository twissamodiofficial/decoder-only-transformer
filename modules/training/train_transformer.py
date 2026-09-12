from modules.transformer.gpt_clone import GPTClone
import torch
import torch.nn as nn
import modules.config as config
import os
import wandb

class TransformerTrainer:
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, attention_module, mlp_ratio=4, device='cpu'):
        self.model = GPTClone(vocab_size, embed_dim, num_heads, num_layers, attention_module, mlp_ratio).to(device)
        self.optimizer = self.set_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.best_val_loss = float('inf')
        self.lr_scheduler = None
        self._pending_scheduler_state = None

    def set_optimizer(self):
        decay_parameters = []
        non_decay_parameters = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() < 2 or "norm" in name.lower() or "bias" in name.lower():
                non_decay_parameters.append(param)
            else:
                decay_parameters.append(param)

        optimizer = torch.optim.AdamW([
            {'params': decay_parameters, 'weight_decay': config.WEIGHT_DECAY},
            {'params': non_decay_parameters, 'weight_decay': 0.0}
        ], lr=config.LEARNING_RATE)
        return optimizer

    def train_step(self, dataloader):
        self.model.train()
        total_loss = 0
        for input_ids, targets in dataloader:
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs, _ = self.model(input_ids)

            loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.GRAD_CLIP_MAX_NORM)  # Gradient clipping
            self.optimizer.step()
            self.lr_scheduler.step()
            total_loss += loss.item()
            
        
        return total_loss / len(dataloader)
    
    def setup_lr_scheduler(self, total_steps):
        warmup_steps = int(0.05 * total_steps)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            total_iters=warmup_steps,
            end_factor=1.0
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    def train(self, train_dataloader, epochs, save_every, val_dataloader=None, start_epoch=0):
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        total_steps = len(train_dataloader) * epochs

        self.setup_lr_scheduler(total_steps)

        if self._pending_scheduler_state is not None:
            self.lr_scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None

        for epoch in range(start_epoch, epochs):
            total_loss = self.train_step(train_dataloader)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss}")

            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                if epoch % 10 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}")

            log_dict = {
                "epoch": epoch,
                "train_loss": total_loss,
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }

            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            wandb.log(log_dict)

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"{config.CHECKPOINT_DIR}/epoch_{epoch+1}.pt", epoch, total_loss, best_val_loss=self.best_val_loss)

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"{config.CHECKPOINT_DIR}/best.pt", epoch, total_loss, best_val_loss=val_loss)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for input_ids, targets in dataloader:
                input_ids = input_ids.to(self.device)
                targets = targets.to(self.device)
                outputs, _ = self.model(input_ids)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)) # crossentropy takes batch_size x seq_len as input
                total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def save_checkpoint(self, path, epoch, loss, best_val_loss=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'train_loss': loss,
            'best_val_loss': best_val_loss
        }, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint.get('best_val_loss', None), checkpoint.get('lr_scheduler_state_dict', None)

    def resume_from_checkpoint(self, path):
        epoch, train_loss, best_val_loss, lr_scheduler_state_dict = self.load_checkpoint(path)
        if best_val_loss is not None:
            self.best_val_loss = best_val_loss
        self._pending_scheduler_state = lr_scheduler_state_dict
        return epoch, train_loss