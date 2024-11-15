# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import SimpleITK as sitk



# 1. Define the Dataset and DataLoader
class MRI2CTDataset(Dataset):
    def __init__(self, data_path_df, transform=None, resize=True):
        self.df = pd.read_csv(data_path_df)
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.df)
    
    def resize_image(self, image, target_size):
        # Resize the 3D image to the target size
        image_resized = nn.functional.interpolate(
            image.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False
        )
        return image_resized.squeeze(0) 

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

        if self.resize:
            mri_image = self.resize_image(mri_image, (28,28,28))
            ct_image = self.resize_image(ct_image, (28,28,28))
        return mri_image, ct_image

def get_dataloader(data_path_df, batch_size=1, transform=None):
    dataset = MRI2CTDataset(data_path_df, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 2. Define a simple one-layer CNN model (replaceable in the future)
class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.decoder = nn.ConvTranspose3d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        # self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        z = self.encoder(x)
        z = torch.relu(z)
        x_hat = self.decoder(z)
        return torch.sigmoid(x_hat)

# 3. Define the Lightning Module
class MRItoCTTranslator(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super(MRItoCTTranslator, self).__init__()
        self.model = model
        self.loss_fn = nn.MSELoss()  # Example loss function for image translation
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mri_image, ct_image = batch
        pred_ct_image = self(mri_image)
        loss = self.loss_fn(pred_ct_image, ct_image)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        mri_image, ct_image = batch
        pred_ct_image = self(mri_image)
        val_loss = self.loss_fn(pred_ct_image, ct_image)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        mri_image, ct_image = batch
        pred_ct_image = self(mri_image)
        test_loss = self.loss_fn(pred_ct_image, ct_image)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

# 4. Training Setup and Execution
# def train_model(mri_data, ct_data, model, batch_size=1, max_epochs=50, learning_rate=1e-3):
    # Get the data loader

train_path_df = "../dataset/train.csv"
val_path_df = "../dataset/val.csv"
test_path_df = "../dataset/test.csv"
transform = None  # Define any necessary transforms
train_dataloader = get_dataloader(train_path_df, batch_size=1, transform=None)
val_dataloader = get_dataloader(val_path_df, batch_size=1, transform=None)
test_dataloader = get_dataloader(test_path_df, batch_size=1, transform=None)


model = EncoderDecoder()
pl_model = MRItoCTTranslator(model, learning_rate=1e-3)

# Define callbacks
checkpoint_callback = ModelCheckpoint(monitor="val_loss")

# Trainer setup
trainer = pl.Trainer(
    max_epochs=2,
    callbacks=[checkpoint_callback],
    log_every_n_steps=1,
)

# Train the model
trainer.fit(pl_model, train_dataloader, val_dataloader)

# Test
trainer.test(pl_model, test_dataloader)