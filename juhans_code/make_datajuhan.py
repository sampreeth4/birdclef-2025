import os
from pathlib import Path
from types import SimpleNamespace
# 필요한 라이브러리 import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
import gc
import pickle as pkl

import librosa

from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
#import torch_audiomentations as tA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler, Adam, AdamW

#import timm

from glob import glob

birdclef_2025_path = r'C:\Users\A.SREE SAI SAMPREETH\Downloads\birdclef-2025'

cfg = SimpleNamespace(**{})
cfg.num_folds = 5
cfg.gpu = "0"

cfg.seed = 2025  # 2024에서 2025로 업데이트

# 설정 업데이트 - 저장 경로를 /kaggle/working으로 변경
cfg.comp_data_path = Path(birdclef_2025_path)  # kagglehub.competition_download('birdclef-2025')의 반환값
cfg.input_path = cfg.comp_data_path  # 또는 필요에 따라 cfg.comp_data_path.parent
cfg.save_path = Path(r'C:\Users\A.SREE SAI SAMPREETH\OneDrive\Documents\GitHub\birdclef-2025\juhans_code')
cfg.output_data_path = cfg.save_path / 'birdclef_data_2025'

os.makedirs(cfg.save_path, exist_ok=True)
os.makedirs(cfg.output_data_path, exist_ok=True)

cfg.logger_file = True

cfg.sr = 32000

cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 먼저 BirdCLEF 2025 데이터셋 구조 확인
print("BirdCLEF 2025 데이터셋 탐색 시작...")
print("Available files in competition directory:")
if os.path.exists(cfg.comp_data_path):
    comp_files = os.listdir(cfg.comp_data_path)
    print(f"Found {len(comp_files)} files/directories: {comp_files}")
else:
    print(f"Competition directory {cfg.comp_data_path} not found!")
    print("Using alternative path...")
    # 대안 경로 시도 (예: 최신 BirdCLEF 디렉토리 찾기)
    possible_dirs = sorted([d for d in os.listdir(cfg.input_path) if 'birdclef' in d.lower()], reverse=True)
    if possible_dirs:
        cfg.comp_data_path = cfg.input_path / possible_dirs[0]
        print(f"Using {cfg.comp_data_path} instead")
        comp_files = os.listdir(cfg.comp_data_path)
        print(f"Found {len(comp_files)} files/directories: {comp_files}")
    else:
        raise FileNotFoundError("Could not find any BirdCLEF data directory")

# train.csv와 taxonomy.csv 로딩 - train_metadata.csv 대신 사용
train = pd.read_csv(cfg.comp_data_path / 'train.csv')
taxonomy = pd.read_csv(cfg.comp_data_path / 'taxonomy.csv')

# 데이터 확인
print("\n데이터 개요:")
print(f"Train data shape: {train.shape} ({train.primary_label.nunique()} unique species)")
print(f"Taxonomy data shape: {taxonomy.shape}")

# 컬럼 비교
print("\n컬럼 비교:")
print(f"Train columns: {train.columns.tolist()}")
print(f"Taxonomy columns: {taxonomy.columns.tolist()}")

# 데이터 샘플 확인
print("\n데이터 샘플:")
print("Train 샘플:")
print(train.head(2))
print("\nTaxonomy 샘플:")
print(taxonomy.head(2))

# train 데이터셋과 taxonomy 결합
print("\n데이터셋 결합...")
train = train.merge(taxonomy[['primary_label', 'class_name']], on='primary_label', how='left')
print(f"Combined data shape: {train.shape}")

# 결측치 확인
missing_class = train[train['class_name'].isna()]
if len(missing_class) > 0:
    print(f"Warning: {len(missing_class)} rows have missing class_name after merge")
    print(f"Missing labels: {missing_class['primary_label'].unique()}")
else:
    print("All species have corresponding taxonomy entries")

# 오디오 파일 로드 함수
def load_audio(filename, cfg):
    # 파일명에서 직접 필요한 정보 추출
    primary_label = None

    # train 데이터프레임에서 해당 파일의 정보 찾기
    file_info = train[train.filename == filename]

    if len(file_info) > 0:
        primary_label = file_info.iloc[0]['primary_label']

        # collection 필드가 있고 값이 있다면 활용
        if 'collection' in file_info.columns and not pd.isna(file_info.iloc[0]['collection']):
            collection = file_info.iloc[0]['collection']
            filepath = cfg.comp_data_path / 'train_audio' / collection / filename
        else:
            # primary_label/filename 형식 사용
            filepath = cfg.comp_data_path / 'train_audio' / primary_label / filename
    else:
        # 파일명에서 primary_label을 추출 (filename이 path 형식이라면)
        if '/' in filename:
            parts = filename.split('/')
            primary_label = parts[0]
            filepath = cfg.comp_data_path / 'train_audio' / filename
        else:
            # 다른 경로 시도
            print(f"Warning: Could not determine path for {filename}, trying direct path")
            filepath = cfg.comp_data_path / 'train_audio' / filename

    # 파일이 존재하는지 확인
    if not os.path.exists(filepath):
        # 대체 경로 시도
        alt_filepath = cfg.comp_data_path / 'train_audio' / filename
        if os.path.exists(alt_filepath):
            filepath = alt_filepath
        else:
            raise FileNotFoundError(f"Could not find audio file at {filepath} or {alt_filepath}")

    audio = librosa.load(filepath, sr=cfg.sr)[0].astype(np.float32)
    return audio

# 테스트를 위한 샘플 파일 로드 - 수정 필요할 수 있음
# 첫 번째 파일로 테스트
sample_filename = train.filename.iloc[0]
try:
    print(f"Loading sample file: {sample_filename}")
    audio = load_audio(sample_filename, cfg)
    print(f"Successfully loaded audio with shape: {audio.shape}")
except Exception as e:
    print(f"Error loading audio: {e}")
    # 파일 경로 구조 확인
    print("Checking file structure...")
    if os.path.exists(cfg.comp_data_path / 'train_audio'):
        print("Found train_audio directory")
        print("Contents:", os.listdir(cfg.comp_data_path / 'train_audio')[:5], "...")
    else:
        print("train_audio directory not found, checking alternatives...")
        print("Root contents:", os.listdir(cfg.comp_data_path))

# 2025년 데이터를 위한 새 디렉토리 생성 (working 디렉토리에)
os.makedirs(cfg.output_data_path, exist_ok=True)
print(f"Created output directory: {cfg.output_data_path}")

# 먼저 train_audio 디렉토리 구조 확인
print("\nChecking train_audio directory structure...")
if os.path.exists(cfg.comp_data_path / 'train_audio'):
    train_audio_contents = os.listdir(cfg.comp_data_path / 'train_audio')
    print(f"Found {len(train_audio_contents)} items in train_audio")
    print(f"Sample items: {train_audio_contents[:5]}")

    # 첫 번째 항목이 디렉토리인지 확인
    first_item_path = cfg.comp_data_path / 'train_audio' / train_audio_contents[0]
    if os.path.isdir(first_item_path):
        print(f"First item '{train_audio_contents[0]}' is a directory containing: {os.listdir(first_item_path)[:3]}")
else:
    print("train_audio directory not found!")

# 데이터 전처리 및 저장
print("\nProcessing audio files...")
dirnames = []
lengths = []
processed_count = 0
error_count = 0
for idx, row in tqdm(train.iterrows(), total=len(train)):
    fname = row.filename
    try:
        audio = load_audio(fname, cfg)

        # 파일명에서 기본 이름만 추출 (확장자 제거)
        if '/' in fname:
            file = fname.split('/')[-1].split('.')[0]
        else:
            file = fname.split('.')[0]

        dirname = row.primary_label  # primary_label을 디렉토리 이름으로 사용
        save_path = cfg.output_data_path / dirname

        # 디렉토리가 없으면 생성
        if dirname not in dirnames:
            os.makedirs(save_path, exist_ok=True)
            dirnames.append(dirname)

        # 짧은 오디오든 긴 오디오든 일관되게 처리
        # 짧은 경우 가능한 만큼만 저장됨
        np.save(save_path / ('first7_' + file), audio[: 7 * cfg.sr])
        np.save(save_path / ('last7_' + file), audio[-7 * cfg.sr : ])
        # 중간 7초 추출 (오디오 길이가 충분할 경우)
        if len(audio) > 14 * cfg.sr:  # 최소 14초 이상인 오디오만 중간 부분 추출
            mid_point = len(audio) // 2
            mid_start = mid_point - int(3.5 * cfg.sr)  # 중간점에서 3.5초 전 (정수로 변환)
            np.save(save_path / ('middle7_' + file), audio[mid_start: mid_start + 7 * cfg.sr])

        lengths.append(audio.shape[0])
        processed_count += 1

        # 진행 상황 출력 (100개마다)
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} files ({error_count} errors)")

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        error_count += 1

# 오디오 길이 히스토그램
plt.figure(figsize=(12, 6))
_ = plt.hist(np.array(lengths) / cfg.sr, bins=100, log=True)
plt.title('Audio Length Distribution (seconds)')
plt.xlabel('Seconds')
plt.ylabel('Count (log scale)')
plt.savefig('audio_length_distribution.png')
plt.close()

# 2024에는 unlabeled_soundscapes가 있었지만 2025에는 없음
print("\nNote: Skipping unlabeled_soundscapes processing as it's not available in BirdCLEF 2025.")

# 특정 종에 대한 데이터 확인
print("\nSample bird species data:")
for label in train.primary_label.unique()[:3]:
    print(f"\nData for {label}:")
    print(train[train.primary_label == label].head(2))

# 결과 요약
print("\nProcessing Summary:")
print(f"Total bird species (primary labels): {train.primary_label.nunique()}")
print(f"Total processed audio files: {len(lengths)}")
print(f"Average audio length: {np.mean(np.array(lengths) / cfg.sr):.2f} seconds")

# 로그 저장
summary_info = {
    'total_species': train.primary_label.nunique(),
    'total_files': len(lengths),
    'avg_audio_length': float(np.mean(np.array(lengths) / cfg.sr)),
    'processed_files': processed_count,
    'error_count': error_count,
    'output_path': str(cfg.output_data_path)
}

# 요약 정보를 JSON으로 저장
with open(cfg.save_path / 'processing_summary.json', 'w') as f:
    import json
    json.dump(summary_info, f, indent=4)

print(f"\nProcessing complete! Files saved to: {cfg.output_data_path}")
print(f"Summary saved to: {cfg.save_path / 'processing_summary.json'}")