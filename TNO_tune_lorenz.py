import os
import csv
import itertools
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import DynamicsDataset
from models.TNO.TNO_lightning import SimpleEncoderModule
from utils import plot_lorenz_prediction_vs_truth
from tslearn.metrics import dtw
from scipy.stats import pearsonr

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
import wandb
from pytorch_lightning.tuner import Tuner

# Import custom modules
from datasets import MetaDataModule
from models.TNO.TNO_lightning import SimpleEncoderModule

# Define your hyperparameter grid
# model_grid = [
#     {'d_model': 32, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 128, 'dropout': 0.1},
#     {'d_model': 64, 'nhead': 8, 'num_layers': 4, 'dim_feedforward': 256, 'dropout': 0.2},
#     # Add more combinations as needed
# ]

param_grid = {
    'd_model': [64, 128, 256],
    'nhead': [4, 8],
    'num_layers': [4, 6],
    'dim_feedforward': [128, 256],    # Create a unique directory for this run's model
}

# Generate all combinations
keys = list(param_grid.keys())
values = list(param_grid.values())
model_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]


data_grid = [
    # Add more combinations as needed
    # {'T': 5, 'batch_size': 64, 'sample_rate': 0.025},
    # {'T': 5, 'batch_size': 128, 'sample_rate': 0.025},
    {'T': 20, 'batch_size': 64, 'sample_rate': 0.025},
    # {'T': 20, 'batch_size': 128, 'sample_rate': 0.025},
    {'T': 50, 'batch_size': 64, 'sample_rate': 0.025},
    # {'T': 50, 'batch_size': 128, 'sample_rate': 0.025}
]

results = []
os.makedirs("plots", exist_ok=True)

for mparams in model_grid:
    for dparams in data_grid:
        # --- 1. Prepare data ---
        dataset = DynamicsDataset(
            size=10000,
            T=dparams['T'],
            sample_rate=dparams['sample_rate'],
            params={'rho': 24.4},
            dyn_sys_name='Lorenz63',
            input_inds=[0],
            output_inds=[1,2],
            test=False
        )
        dataloader = DataLoader(dataset, batch_size=dparams['batch_size'], shuffle=True)

        # --- 2. Build model ---
        model = SimpleEncoderModule(
            input_dim=1,
            output_dim=2,
            domain_dim=1,
            d_model=mparams['d_model'],
            nhead=mparams['nhead'],
            num_layers=mparams['num_layers'],
            dim_feedforward=mparams['dim_feedforward'],
            dropout=1e-4,
            learning_rate=1e-3,
            activation='gelu'
        )

        # --- 3. Train model (simple loop, or use Trainer as in notebook) ---
        # For brevity, here we skip training code; insert your Trainer.fit() here
        # Set up callbacks (optional but recommended)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        early_stop = EarlyStopping(monitor='loss/val/mse', patience=10, mode="min")

        # Set up PyTorch Lightning Trainer (no WandbLogger)
        trainer = Trainer(
            max_epochs=200,
            callbacks=[lr_monitor, early_stop],
            accelerator="auto",  # use GPU if available
            strategy='ddp_find_unused_parameters_true',
        )

        # Fit the model using the dataloader
        trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
        

        # --- 4. Evaluate and plot ---
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        with torch.no_grad():
            for x_batch, y_batch, times_x, times_y in dataloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                times_x = times_x.to(device)
                times_y = times_y.to(device)
                y_pred = model(x_batch, y_batch, times_x, times_y)
                break  # Only first batch

        # Create a unique directory for this run
        run_name = (
            f"dmodel{mparams['d_model']}_nhead{mparams['nhead']}_layers{mparams['num_layers']}_"
            f"ff{mparams['dim_feedforward']}_T{dparams['T']}_bs{dparams['batch_size']}_sr{dparams['sample_rate']}"
        )
        run_dir = os.path.join("plots", run_name)
        os.makedirs(run_dir, exist_ok=True)

        # Save plots and errors for each sample in batch
        for i in range(x_batch.shape[0]):
            t_i = times_y[i].cpu().numpy()
            x_true_i = x_batch[i, :, 0].cpu().numpy()
            y_true_i = y_batch[i, :, 0].cpu().numpy()
            z_true_i = y_batch[i, :, 1].cpu().numpy()
            y_pred_i = y_pred[i, :, 0].cpu().numpy()
            z_pred_i = y_pred[i, :, 1].cpu().numpy()

            # --- Compute DTW and Pearson r ---
            dtw_y = dtw(y_true_i, y_pred_i)
            dtw_z = dtw(z_true_i, z_pred_i)
            pearson_y, _ = pearsonr(y_true_i, y_pred_i)
            pearson_z, _ = pearsonr(z_true_i, z_pred_i)

            # Save plot
            plot_path = os.path.join(run_dir, f"sample{i}.png")
            plot_lorenz_prediction_vs_truth(
                t_i, x_true_i, y_true_i, z_true_i, y_pred_i, z_pred_i,
                title_prefix=f"Sample {i}: ",
                model_hparams=f"{mparams}, {dparams}",
                save_path=plot_path
            )

            # Save results
            result = {
                **mparams,
                **dparams,
                'sample': i,
                'dtw_y': dtw_y,
                'dtw_z': dtw_z,
                'pearson_y': pearson_y,
                'pearson_z': pearson_z,
                'plot_path': plot_path
            }
            results.append(result)

        # Save the model checkpoint with hyperparameters in the filename
        model_filename = (
            f"model_dmodel{mparams['d_model']}_nhead{mparams['nhead']}_layers{mparams['num_layers']}_"
            f"ff{mparams['dim_feedforward']}_T{dparams['T']}_bs{dparams['batch_size']}_sr{dparams['sample_rate']}.pt"
        )
        model_save_path = os.path.join(run_dir, model_filename)
        torch.save(model.state_dict(), model_save_path)

        # Optionally, clear memory
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()

# --- 5. Save all results to CSV ---
with open("lorenz_hyperparam_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)