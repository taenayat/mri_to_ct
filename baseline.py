# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import pandas as pd
import SimpleITK as sitk



# 1. Define the Dataset and DataLoader
class MRI2CTDataset(Dataset):
    def __init__(self, data_path_df, transform=None):
        self.df = pd.read_csv(data_path_df)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        mri_path = self.df.iloc[idx]['mri_paths']
        ct_path = self.df.iloc[idx]['ct_paths']
        mask_path = self.df.iloc[idx]['mask_paths']

        mri_image_np = sitk.GetArrayFromImage(sitk.ReadImage(mri_path))
        ct_image_np = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        mask_image_np = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))

        mri_image = torch.tensor(mri_image_np, dtype=torch.float32).unsqueeze(0)
        ct_image = torch.tensor(ct_image_np, dtype=torch.float32).unsqueeze(0)
        mask_image = torch.tensor(mask_image_np, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            mri_image = self.transform(mri_image)
            ct_image = self.transform(ct_image)
        return mri_image, ct_image

def get_dataloader(mri_data, ct_data, batch_size=1, transform=None):
    dataset = MRI2CTDataset(mri_data, ct_data, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Define a simple one-layer CNN model (replaceable in the future)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

# 3. Define the Lightning Module
class MRItoCTTranslator(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(MRItoCTTranslator, self).__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()  # Example loss function for image translation
        self.learning_rate = learning_rate

    def forward(self, x):
        # Forward pass
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Single training step
        mri_image, ct_image = batch
        pred_ct_image = self(mri_image)
        loss = self.loss_fn(pred_ct_image, ct_image)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        # Define optimizer and learning rate scheduler
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "train_loss"}

# 4. Training Setup and Execution
def train_model(mri_data, ct_data, model, batch_size=1, max_epochs=50, learning_rate=1e-3):
    # Get the data loader
    transform = None  # Define any necessary transforms
    dataloader = get_dataloader(mri_data, ct_data, batch_size, transform)

    # Initialize model
    translator_model = MRItoCTTranslator(model, learning_rate)

    # Define callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(monitor="train_loss")

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=10,
    )

    # Train the model
    trainer.fit(translator_model, dataloader)

# Usage Example
# Assuming you have MRI and CT data loaded as mri_data and ct_data
model = SimpleCNN()
train_model(mri_data, ct_data, model, batch_size=32, max_epochs=50, learning_rate=1e-3)
