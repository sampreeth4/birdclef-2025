import torch
import os
import pandas as pd
import librosa
import numpy as np
from model import BirdCLEFModel  # same model definition
from config import cfg  # your CFG class
from tqdm import tqdm

# Load model
model = BirdCLEFModel()
model.load_state_dict(torch.load('efficientnet_b0_birdclef.pth', map_location=cfg.device))
model.to(cfg.device)
model.eval()

# Prepare submission
test_files = os.listdir('path_to_test_audio')  # test_audio directory
results = []

for fname in tqdm(test_files):
    path = os.path.join('path_to_test_audio', fname)
    y, _ = librosa.load(path, sr=cfg.sr)
    if len(y) < cfg.sr * cfg.duration:
        y = np.pad(y, (0, cfg.sr * cfg.duration - len(y)))
    else:
        y = y[:cfg.sr * cfg.duration]
    
    mel = librosa.feature.melspectrogram(
        y, sr=cfg.sr, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, hop_length=cfg.hop_length)
    mel = librosa.power_to_db(mel).astype(np.float32)
    mel -= mel.mean()
    mel /= mel.std()
    mel = np.expand_dims(mel, axis=0)  # (1, H, W)
    mel_tensor = torch.tensor(mel).unsqueeze(0).to(cfg.device)  # (1, 1, H, W)

    with torch.no_grad():
        pred = torch.sigmoid(model(mel_tensor)).cpu().numpy()[0]

    # For submission: pick top prediction(s)
    top_label = np.argmax(pred)
    label = list(label_map.keys())[list(label_map.values()).index(top_label)]
    
    results.append({"row_id": fname.replace('.ogg', ''), "primary_label": label})

# Save CSV
submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)
