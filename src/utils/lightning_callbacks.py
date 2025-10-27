import pytorch_lightning as pl
import torch
import os
import pickle
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import wandb
import time


class SaveIntermediatesCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        
    def save_intermediates(self, base_dir, trainer, pl_module, batch):
        os.makedirs(base_dir, exist_ok=True)
        
        pickle.dump(
            pl_module._last_intermediates,
            open(os.path.join(base_dir,  "intermediates.pkl"), "wb")
        )
        trainer.save_checkpoint(os.path.join(base_dir, "model.ckpt"))
        torch.save(batch, os.path.join(base_dir, "batch.pt"))
        

class SaveBeforeNaNCallback(SaveIntermediatesCallback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Check parameters and gradients for NaNs
        nan_found = False
        for name, p in pl_module.named_parameters():
            if torch.isnan(p).any():
                print(f"NaN detected in parameter {name} at step {trainer.global_step}")
                nan_found = True
                break
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"NaN detected in gradient {name} at step {trainer.global_step}")
                nan_found = True
                break

        if nan_found:
            base_dir = os.path.join(self.save_path, f"nan_params_{trainer.global_step}")
            self.save_intermediates(base_dir, trainer, pl_module, batch)
            trainer.should_stop = True
            

class SaveHighLossesCallback(SaveIntermediatesCallback):
    def __init__(self, save_path, loss_threshold=30.0):
        super().__init__()
        self.save_path = save_path
        self.loss_threshold = loss_threshold

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs["loss"]
        if loss > self.loss_threshold:
            print(f"High loss detected: {loss:.2f} at step {trainer.global_step}")
            base_dir = os.path.join(self.save_path, f"high_loss_{trainer.global_step}")
            self.save_intermediates(base_dir, trainer, pl_module, batch)
            
            with open(os.path.join(base_dir, "debug_info.txt"), "w") as f:
                f.write(self.get_debug_text(pl_module._last_intermediates))
    
    def get_debug_text(self, intermediates):
            save_text = ""
            t = intermediates["t"]
            tau = intermediates["tau"]
            m = intermediates["m"]
            x_t = intermediates["x_t"]
            x = intermediates["x_0"]
            predicted_x0 = intermediates["predicted_x0"]
            loss = intermediates["loss"]

            # bad loss
            # B x D x C
            b = loss.mean(dim=1).argmax().item()
            d = loss[b].argmax().item()
            save_text += f"worst loss: b={b}, d={d}, loss={loss[b, d].item()}\n"

            bad_t = t[b]
            bad_m = m[b, d] if m is not None else None
            bad_tau = tau[b]
            bad_x_t = x_t[b, d].cpu().numpy()
            bad_x = x[b, d].cpu().numpy()
            pred_x0 = predicted_x0[b, d].cpu().numpy()
            
            
            # highlight in pred_x0:
            argmax_idx = np.argmax(pred_x0)
            
            def pred_formatter_x(v):
                if np.isclose(v, bad_x_t[argmax_idx]):  # argmax entry
                    return f"!!!{v:.4f}!!!"
                elif np.isclose(v, bad_x_t[bad_x]):       # bad_x entry
                    return f"-->{v:.4f}<--"
                else:
                    return f"{v:.4f}"
                
            highlighted_x_t = np.array2string(
                bad_x_t,
                formatter={'float_kind': pred_formatter_x}
            )

            def pred_formatter_pred(v):
                if np.isclose(v, pred_x0[argmax_idx]):  # argmax entry
                    return f"!!!{v:.4f}!!!"
                elif np.isclose(v, pred_x0[bad_x]):       # bad_x entry
                    return f"-->{v:.4f}<--"
                else:
                    return f"{v:.4f}"
                

            highlighted_pred_x0 = np.array2string(
                pred_x0,
                formatter={'float_kind': pred_formatter_pred}
            )
            
            
            save_text += f"bad_t: {bad_t}, bad_m: {bad_m}, bad_tau: {bad_tau}\n"
            save_text += f"original x: {bad_x}\n"
            
            save_text += f"noisy x_t:\n{highlighted_x_t}\n"
            save_text += f"predicted x0 (argmax at {argmax_idx}):\n{highlighted_pred_x0}\n"
            
            return save_text
            

class DumpProfileCallback(pl.Callback):
    def __init__(self, save_path, profiler, every_n_steps=5):
        self.profiler = profiler
        self.every_n_steps = every_n_steps
        self.save_path = save_path

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_steps == 0 and self.profiler is not None:
            # write to file
            with open(os.path.join(self.save_path, f"profiler_summary.txt"), "w") as f:
                f.write("Profiler summary at step {}\n".format(trainer.global_step))
                f.write(self.profiler.summary())
                f.write("\n\n")
                
                
class UpdateEpochCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module, "train_dataloader") and hasattr(pl_module.train_dataloader.dataset, 'set_epoch'):
            pl_module.train_dataloader.dataset.set_epoch(trainer.current_epoch)
            

    def on_validation_epoch_start(self, trainer, pl_module):
        if hasattr(pl_module, "val_dataloader") and hasattr(pl_module.val_dataloader.dataset, 'set_epoch'):
            pl_module.val_dataloader.dataset.set_epoch(trainer.current_epoch)
            
            
class VisualizeLossVsTimeCallback(pl.Callback):
    def __init__(self, every_n_steps=100):
        super().__init__()
        self.every_n_steps = every_n_steps
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (wandb.run.step % self.every_n_steps) != 0:
            return
        t = pl_module._last_intermediates["t"]
        loss = pl_module._last_intermediates["loss"]

        t_list = t.detach().cpu().tolist()
        loss_list = loss.detach().cpu().mean(1).tolist()
        wandb_table = wandb.Table(data=list(zip(t_list, loss_list)), columns=["t", "loss"])
        wandb.log({
            "loss_vs_t": wandb_table,
            "scatter_loss_vs_t": wandb.plot.scatter(wandb_table, "t", "loss", title="Loss vs t"),
        }, step=wandb.run.step)

class TimingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.step_start_time = None
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # This runs RIGHT BEFORE your training_step method is called
        self.step_start_time = time.time()
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # This runs AFTER the complete training step (including backprop) finishes
        if self.step_start_time is not None:
            total_step_time = (time.time() - self.step_start_time) * 1000
            # Log the timing to your model's logger
            pl_module.log("total_step_time_ms", total_step_time)