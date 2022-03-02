import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1, delta=0,trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.trace_func=trace_func
        self.patience = patience
        
        self.early_stop = False
        self.delta = delta

    def __call__(self,bestloss, val_loss, model, path,counter):
        self.path=path
        self.best_val_loss=bestloss
        self.counter = counter
        score = val_loss

        if self.best_val_loss >score:
            print(f"New best model for val loss : {score:4.4}! saving the best model..")
            torch.save(model.module.state_dict(), f"{self.path}/best.pth")
            self.best_val_loss = score
        elif score > self.best_val_loss + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            torch.save(model.module.state_dict(), f"{self.path}/last.pth")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = score
            torch.save(model.module.state_dict(), f"{self.path}/best.pth")
            self.counter = 0
        return self.best_val_loss,self.counter
