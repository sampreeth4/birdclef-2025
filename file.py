import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchaudio
import librosa
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import timm

# CONFIG
class CFG:
    sr = 32000  # Sample rate
    duration = 5  # seconds
    n_mels = 128
    fmin = 20
    fmax = sr // 2
    hop_length = 512
    batch_size = 32
    num_workers = 4
    model_name = 'efficientnet_b0'
    num_classes = 206
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = CFG()

# Dataset
class BirdDataset(Dataset):
    def __init__(self, df, audio_dir, transform=None):
        self.df = df
        self.audio_dir = audio_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.audio_dir, row["filename"])
        
        # Load the full audio
        y, sr = librosa.load(file_path, sr=cfg.sr)

        # Random 5 second clip
        clip_samples = cfg.sr * cfg.duration
        if len(y) < clip_samples:
            padding = clip_samples - len(y)
            y = np.pad(y, (0, padding))
        else:
            start_idx = np.random.randint(0, len(y) - clip_samples)
            y = y[start_idx:start_idx + clip_samples]

        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y, sr=sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, hop_length=cfg.hop_length
        )
        mel_spec = librosa.power_to_db(mel_spec).astype(np.float32)

        # Normalize
        mel_spec -= mel_spec.mean()
        mel_spec /= mel_spec.std()

        # (Channels, Height, Width)
        mel_spec = np.expand_dims(mel_spec, axis=0)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        label = np.zeros(cfg.num_classes, dtype=np.float32)
        label[row["primary_label_idx"]] = 1.0

        return torch.tensor(mel_spec), torch.tensor(label)

# Model
class BirdCLEFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(cfg.model_name, pretrained=True, in_chans=1)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(n_features, cfg.num_classes)

    def forward(self, x):
        return self.backbone(x)

# Prepare train data
train_df = pd.read_csv('C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/train.csv')  # TODO: set correct path

# Map primary labels to 0-205
label_map = {label: idx for idx, label in enumerate(sorted(train_df["primary_label"].unique()))}
train_df["primary_label_idx"] = train_df["primary_label"].map(label_map)

train_dataset = BirdDataset(train_df, 'C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/train_audio')  # TODO: set correct path
train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

# Train
model = BirdCLEFModel().to(cfg.device)

# Freeze all except head
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.backbone.classifier.parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for inputs, targets in pbar:
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix(loss=running_loss / (pbar.n + 1))

# Save model
torch.save(model.state_dict(), 'efficientnet_b0_birdclef.pth')
