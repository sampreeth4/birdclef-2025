import os
import logging
import random
import gc
import time
import cv2
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import timm
from scipy import ndimage
import time
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

class CFG:
    
    seed = 42
    debug = False  
    apex = False
    print_freq = 100
    num_workers = 2
    
    OUTPUT_DIR = 'C:/Users/A.SREE SAI SAMPREETH/OneDrive/Documents/GitHub/birdclef-2025/working'

    train_datadir = 'C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/train_audio'
    train_csv = 'C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/train.csv'
    test_soundscapes = 'C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/test_soundscapes'
    submission_csv = 'C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/sample_submission.csv'
    taxonomy_csv = 'C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/taxonomy.csv'

    # 경로 변경: 10초 데이터가 저장된 폴더 위치로 수정
    spectrogram_npy = None  # npy 파일 하나가 아닌 폴더 기반으로 변경
    audio_data_dir = "C:/Users/A.SREE SAI SAMPREETH/OneDrive/Documents/GitHub/birdclef-2025/juhans_code/birdclef_data_2025"  # 10초 세그먼트 데이터 폴더

    # 모델명 변경
    model_name = 'tf_efficientnet_b0.ns_jft_in1k'  # efficientvit_b1, regnety_008, efficientnet_b1, efficientvit_b0, efficientnet_b0, convnextv2_tiny, convnext2_base, convnextv2_atto, lcnet_050 등 사용 가능
    pretrained = True
    in_channels = 1

    LOAD_DATA = True  
    FS = 32000
    TARGET_DURATION = 5.0
    TARGET_SHAPE = (256, 256)
    #TARGET_SHAPE = (224, 224)
    
    N_FFT = 4096
    HOP_LENGTH = 512
    N_MELS = 512
    FMIN = 0
    FMAX = 16000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 100  
    batch_size = 64  
    #criterion = 'BCEWithLogitsLoss'
    criterion = 'BCEFocalLoss'
    #criterion = 'FocalLoss'

    n_fold = 5
    selected_folds = [0, 1, 2, 3, 4]   

    optimizer = 'AdamW'
    lr = 5e-4 
    weight_decay = 1e-5
  
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = epochs

    aug_prob = 0.5  
    mixup_alpha = 0.5  

    # 새로운 증강 파라미터 추가
    use_enhanced_augmentations = True  # 향상된 증강 사용 여부
    p_horizontal_flip = 0.5  # HorizontalFlip 확률
    p_coarse_dropout = 0.5  # CoarseDropout 확률
    p_local_stretch = 0.5  # Local Time-Frequency Stretching 확률
    p_global_stretch = 0.5  # Global Time-Frequency Stretching 확률
    p_amplitude_scaling = 0.5  # Amplitude Scaling 확률
    p_noise_injection = 0.5  # Noise Injection 확률

    focal_alpha = 0.5
    focal_gamma = 2.0
    
    def update_debug_settings(self):
        if self.debug:
            self.epochs = 2
            self.selected_folds = [0]

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def audio2melspec(audio_data, cfg):
    """Convert audio data to mel spectrogram"""
    if np.isnan(audio_data).any():
        mean_signal = np.nanmean(audio_data)
        audio_data = np.nan_to_num(audio_data, nan=mean_signal)

    mel_spec = librosa.feature.melspectrogram(
        y=audio_data,
        sr=cfg.FS,
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        power=2.0
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm

def process_audio_file(audio_path, cfg):
    """Process a single audio file to get the mel spectrogram"""
    try:
        audio_data, _ = librosa.load(audio_path, sr=cfg.FS)

        target_samples = int(cfg.TARGET_DURATION * cfg.FS)

        if len(audio_data) < target_samples:
            n_copy = math.ceil(target_samples / len(audio_data))
            if n_copy > 1:
                audio_data = np.concatenate([audio_data] * n_copy)

        # Extract center 5 seconds
        start_idx = max(0, int(len(audio_data) / 2 - target_samples / 2))
        end_idx = min(len(audio_data), start_idx + target_samples)
        center_audio = audio_data[start_idx:end_idx]

        if len(center_audio) < target_samples:
            center_audio = np.pad(center_audio, 
                                 (0, target_samples - len(center_audio)), 
                                 mode='constant')

        mel_spec = audio2melspec(center_audio, cfg)
        
        if mel_spec.shape != cfg.TARGET_SHAPE:
            mel_spec = cv2.resize(mel_spec, cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)

        return mel_spec.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_spectrograms(df, cfg):
    """Generate spectrograms from audio files"""
    print("Generating mel spectrograms from audio files...")
    start_time = time.time()

    all_bird_data = {}
    errors = []

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if cfg.debug and i >= 1000:
            break
        
        try:
            samplename = row['samplename']
            filepath = row['filepath']
            
            mel_spec = process_audio_file(filepath, cfg)
            
            if mel_spec is not None:
                all_bird_data[samplename] = mel_spec
            
        except Exception as e:
            print(f"Error processing {row.filepath}: {e}")
            errors.append((row.filepath, str(e)))

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_bird_data)} files out of {len(df)}")
    print(f"Failed to process {len(errors)} files")
    
    return all_bird_data

class AudioAugmentations:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # 각 증강 기법의 적용 확률 설정
        self.p_horizontal_flip = 0.5
        self.p_coarse_dropout = 0.5
        self.p_local_stretch = 0.5
        self.p_global_stretch = 0.5
        self.p_amplitude_scaling = 0.5
        self.p_noise_injection = 0.5
        
        # 기존 증강 확률
        self.p_time_mask = 0.5
        self.p_freq_mask = 0.5
        self.p_brightness_contrast = 0.5
        
        # 새로운 증강 기법 확률 추가
        self.p_random_erasing = 0.5
        self.p_rand_augment = 0.5
        
        # 증강 강도 설정
        self.noise_level = 0.05
        self.amplitude_range = (0.1, 1.6) # 작년 2등은 (0.1, 1.6)
        self.global_stretch_range = (0.8, 1.2)
        self.local_stretch_max_percent = 0.2
        self.coarse_dropout_size_range = (10, 40)
        self.coarse_dropout_count = 4
        
        # Random Erasing 설정
        self.random_erasing_scale = (0.02, 0.1)  # 지울 영역의 크기 비율
        self.random_erasing_ratio = (0.3, 3.3)    # 지울 영역의 가로/세로 비율
        self.random_erasing_value = 0             # 지워진 영역을 채울 값
        
        # RandAugment 설정
        self.rand_augment_n = 2  # 적용할 변환의 수
        self.rand_augment_m = 9  # 변환의 강도 (1-10)
    
    def apply_all_augmentations(self, spec):
        """모든 증강 기법을 확률적으로 적용"""
        # 원본 스펙트로그램 복사
        augmented_spec = spec.clone()
        
        # 기존 증강 기법 적용
        if random.random() < self.p_time_mask:
            augmented_spec = self.apply_time_masking(augmented_spec)
        
        if random.random() < self.p_freq_mask:
            augmented_spec = self.apply_freq_masking(augmented_spec)
        
        if random.random() < self.p_brightness_contrast:
            augmented_spec = self.apply_random_brightness_contrast(augmented_spec)
        
        # 새로운 증강 기법 적용
        if random.random() < self.p_horizontal_flip:
            augmented_spec = self.apply_horizontal_flip(augmented_spec)
        
        if random.random() < self.p_coarse_dropout:
            augmented_spec = self.apply_coarse_dropout(augmented_spec)
        
        if random.random() < self.p_local_stretch:
            augmented_spec = self.apply_local_time_freq_stretch(augmented_spec)
        
        if random.random() < self.p_global_stretch:
            augmented_spec = self.apply_global_time_freq_stretch(augmented_spec)
        
        if random.random() < self.p_amplitude_scaling:
            augmented_spec = self.apply_amplitude_scaling(augmented_spec)
        
        if random.random() < self.p_noise_injection:
            augmented_spec = self.apply_noise_injection(augmented_spec)
            
        # 새로운 증강 기법 적용
        if random.random() < self.p_random_erasing:
            augmented_spec = self.apply_random_erasing(augmented_spec)
        
        if random.random() < self.p_rand_augment:
            augmented_spec = self.apply_rand_augment(augmented_spec)
        
        return augmented_spec
    
    def apply_time_masking(self, spec):
        """시간 마스킹 증강"""
        spec_aug = spec.clone()
        num_masks = random.randint(1, 3)
        for _ in range(num_masks):
            width = random.randint(5, 20)
            start = random.randint(0, spec.shape[2] - width)
            spec_aug[0, :, start:start+width] = 0
        return spec_aug
    
    def apply_freq_masking(self, spec):
        """주파수 마스킹 증강"""
        spec_aug = spec.clone()
        num_masks = random.randint(1, 3)
        for _ in range(num_masks):
            height = random.randint(5, 20)
            start = random.randint(0, spec.shape[1] - height)
            spec_aug[0, start:start+height, :] = 0
        return spec_aug
    
    def apply_random_brightness_contrast(self, spec):
        """밝기/대비 조정 증강"""
        spec_aug = spec.clone()
        gain = random.uniform(0.8, 1.2)
        bias = random.uniform(-0.1, 0.1)
        spec_aug = spec_aug * gain + bias
        spec_aug = torch.clamp(spec_aug, 0, 1)
        return spec_aug
    
    def apply_horizontal_flip(self, spec):
        """스펙트로그램 수평 뒤집기 (시간축 반전)"""
        spec_aug = spec.clone()
        spec_aug = torch.flip(spec_aug, [2])  # 시간 차원으로 뒤집기
        return spec_aug
    
    def apply_coarse_dropout(self, spec):
        """CoarseDropout: 큰 사각형 영역을 랜덤하게 마스킹"""
        spec_aug = spec.clone()
        for _ in range(self.coarse_dropout_count):
            size_h = random.randint(*self.coarse_dropout_size_range)
            size_w = random.randint(*self.coarse_dropout_size_range)
            
            # 마스크 시작 위치 랜덤 선택
            if spec.shape[1] > size_h and spec.shape[2] > size_w:
                h_start = random.randint(0, spec.shape[1] - size_h)
                w_start = random.randint(0, spec.shape[2] - size_w)
                
                # 마스킹 적용
                spec_aug[0, h_start:h_start+size_h, w_start:w_start+size_w] = 0
        
        return spec_aug
    
    def apply_local_time_freq_stretch(self, spec):
        """국소적 시간-주파수 스트레칭: 스펙트로그램의 일부 영역만 늘리거나 줄임"""
        # 텐서를 numpy로 변환
        spec_np = spec[0].cpu().numpy()
        height, width = spec_np.shape
        
        # 변형할 영역 선택 (전체의 약 30-70% 영역)
        region_h = int(height * random.uniform(0.3, 0.7))
        region_w = int(width * random.uniform(0.3, 0.7))
        h_start = random.randint(0, height - region_h)
        w_start = random.randint(0, width - region_w)
        
        # 선택 영역 추출
        region = spec_np[h_start:h_start+region_h, w_start:w_start+region_w]
        
        # 변형 비율 결정
        stretch_h = random.uniform(1.0 - self.local_stretch_max_percent, 
                                 1.0 + self.local_stretch_max_percent)
        stretch_w = random.uniform(1.0 - self.local_stretch_max_percent, 
                                 1.0 + self.local_stretch_max_percent)
        
        # 리사이징 적용
        new_h = max(1, int(region_h * stretch_h))
        new_w = max(1, int(region_w * stretch_w))
        
        # 리사이징된 영역
        resized_region = cv2.resize(region, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 원본 크기에 맞게 재조정
        if new_h != region_h or new_w != region_w:
            resized_region = cv2.resize(resized_region, (region_w, region_h), interpolation=cv2.INTER_LINEAR)
            
        # 변형된 영역을 원본에 적용
        spec_np[h_start:h_start+region_h, w_start:w_start+region_w] = resized_region
        
        # numpy를 다시 텐서로 변환
        spec_aug = spec.clone()
        spec_aug[0] = torch.from_numpy(spec_np)
        
        return spec_aug
    
    def apply_global_time_freq_stretch(self, spec):
        """전역 시간-주파수 스트레칭: 전체 스펙트로그램 늘리거나 줄임"""
        # 텐서를 numpy로 변환
        spec_np = spec[0].cpu().numpy()
        original_shape = spec_np.shape
        
        # 변형 비율 결정
        stretch_h = random.uniform(*self.global_stretch_range)
        stretch_w = random.uniform(*self.global_stretch_range)
        
        # 새 크기 계산
        new_h = max(1, int(original_shape[0] * stretch_h))
        new_w = max(1, int(original_shape[1] * stretch_w))
        
        # 리사이징 적용
        resized = cv2.resize(spec_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 원본 크기로 다시 조정
        if new_h != original_shape[0] or new_w != original_shape[1]:
            resized = cv2.resize(resized, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # numpy를 다시 텐서로 변환
        spec_aug = spec.clone()
        spec_aug[0] = torch.from_numpy(resized)
        
        return spec_aug
    
    def apply_amplitude_scaling(self, spec):
        """진폭 스케일링: 스펙트로그램의 값 범위를 변경"""
        spec_aug = spec.clone()
        scale_factor = random.uniform(*self.amplitude_range)
        spec_aug = spec_aug * scale_factor
        spec_aug = torch.clamp(spec_aug, 0, 1)  # 0-1 범위로 클리핑
        return spec_aug
    
    def apply_noise_injection(self, spec):
        """노이즈 주입: 랜덤 노이즈를 스펙트로그램에 추가"""
        spec_aug = spec.clone()
        noise = torch.randn_like(spec[0]) * self.noise_level
        spec_aug[0] = spec_aug[0] + noise
        spec_aug = torch.clamp(spec_aug, 0, 1)  # 0-1 범위로 클리핑
        return spec_aug
    
    def apply_random_erasing(self, spec):
        """RandomErasing: 이미지의 랜덤한 작은 영역을 완전히 지우는 기법"""
        spec_aug = spec.clone()
        
        height = spec.shape[1]
        width = spec.shape[2]
        area = height * width
        
        for _ in range(random.randint(1, 3)):  # 1~3개의 영역 지우기
            for attempt in range(10):  # 10번 시도
                target_area = random.uniform(*self.random_erasing_scale) * area
                aspect_ratio = random.uniform(*self.random_erasing_ratio)
                
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                
                if w < width and h < height:
                    x = random.randint(0, width - w)
                    y = random.randint(0, height - h)
                    
                    spec_aug[0, y:y+h, x:x+w] = self.random_erasing_value
                    break
        
        return spec_aug
    
    def apply_rand_augment(self, spec):
        """RandAugment: 여러 증강 기법을 랜덤하게 조합하는 방식"""
        spec_aug = spec.clone()
        
        # 가능한 변환 목록
        transforms = [
            self._rotate,
            self._translate_x,
            self._translate_y,
            self._shear_x,
            self._shear_y,
            self._contrast,
            self._sharpness,
            self._cutout
        ]
        
        # N개의 변환을 랜덤하게 선택하여 적용
        selected_transforms = random.sample(transforms, min(self.rand_augment_n, len(transforms)))
        
        for transform_fn in selected_transforms:
            # 각 변환에 강도 M 적용
            magnitude = self.rand_augment_m / 10  # 0.1 ~ 1.0 범위로 정규화
            spec_aug = transform_fn(spec_aug, magnitude)
        
        return spec_aug
    
    # RandAugment에 사용할 개별 변환 함수들
    def _rotate(self, spec, magnitude):
        """이미지 회전"""
        # Numpy로 변환
        spec_np = spec[0].cpu().numpy()
        
        # 회전 각도 계산 (-30도에서 30도 사이)
        angle = magnitude * 30.0
        if random.random() < 0.5:
            angle = -angle
        
        # 회전 변환
        rotated = ndimage.rotate(spec_np, angle, reshape=False, mode='constant', cval=0)
        
        # 텐서로 다시 변환
        spec_aug = spec.clone()
        spec_aug[0] = torch.from_numpy(rotated.astype(np.float32))
        spec_aug = torch.clamp(spec_aug, 0, 1)
        
        return spec_aug
    
    def _translate_x(self, spec, magnitude):
        """X축(시간) 방향으로 이동"""
        spec_aug = spec.clone()
        pixels = int(magnitude * spec.shape[2] * 0.45)
        if random.random() < 0.5:
            pixels = -pixels
        
        # 이동 적용
        if pixels > 0:
            spec_aug[0, :, pixels:] = spec[0, :, :-pixels]
            spec_aug[0, :, :pixels] = 0
        elif pixels < 0:
            pixels = abs(pixels)
            spec_aug[0, :, :-pixels] = spec[0, :, pixels:]
            spec_aug[0, :, -pixels:] = 0
        
        return spec_aug
    
    def _translate_y(self, spec, magnitude):
        """Y축(주파수) 방향으로 이동"""
        spec_aug = spec.clone()
        pixels = int(magnitude * spec.shape[1] * 0.45)
        if random.random() < 0.5:
            pixels = -pixels
        
        # 이동 적용
        if pixels > 0:
            spec_aug[0, pixels:, :] = spec[0, :-pixels, :]
            spec_aug[0, :pixels, :] = 0
        elif pixels < 0:
            pixels = abs(pixels)
            spec_aug[0, :-pixels, :] = spec[0, pixels:, :]
            spec_aug[0, -pixels:, :] = 0
        
        return spec_aug
    
    def _shear_x(self, spec, magnitude):
        """X축 방향으로 기울이기"""
        # Numpy로 변환
        spec_np = spec[0].cpu().numpy()
        
        # 기울기 각도 계산
        shear_factor = magnitude * 0.5  # 최대 0.5
        if random.random() < 0.5:
            shear_factor = -shear_factor
        
        # 변환 행렬 생성
        height, width = spec_np.shape
        afine_tf = np.array([[1, shear_factor, 0], [0, 1, 0]])
        
        # 어파인 변환 적용
        sheared = cv2.warpAffine(spec_np, afine_tf, (width, height), borderValue=0)
        
        # 텐서로 다시 변환
        spec_aug = spec.clone()
        spec_aug[0] = torch.from_numpy(sheared.astype(np.float32))
        
        return spec_aug
    
    def _shear_y(self, spec, magnitude):
        """Y축 방향으로 기울이기"""
        # Numpy로 변환
        spec_np = spec[0].cpu().numpy()
        
        # 기울기 각도 계산
        shear_factor = magnitude * 0.5  # 최대 0.5
        if random.random() < 0.5:
            shear_factor = -shear_factor
        
        # 변환 행렬 생성
        height, width = spec_np.shape
        afine_tf = np.array([[1, 0, 0], [shear_factor, 1, 0]])
        
        # 어파인 변환 적용
        sheared = cv2.warpAffine(spec_np, afine_tf, (width, height), borderValue=0)
        
        # 텐서로 다시 변환
        spec_aug = spec.clone()
        spec_aug[0] = torch.from_numpy(sheared.astype(np.float32))
        
        return spec_aug
    
    def _contrast(self, spec, magnitude):
        """대비 조정"""
        spec_aug = spec.clone()
        
        # 대비 계산 (1.0은 변화 없음, 2.0은 두 배 증가, 0.5는 절반 감소)
        factor = 1.0 + magnitude * 1.0  # 1.0 ~ 2.0 범위
        
        # 평균값 계산
        mean = torch.mean(spec_aug)
        
        # 대비 적용
        spec_aug = (spec_aug - mean) * factor + mean
        spec_aug = torch.clamp(spec_aug, 0, 1)
        
        return spec_aug
    
    def _sharpness(self, spec, magnitude):
        """선명도 조정 (가우시안 블러의 반대 적용)"""
        # Numpy로 변환
        spec_np = spec[0].cpu().numpy()
        
        # 원본 복사
        blurred = cv2.GaussianBlur(spec_np, (5, 5), 1.0)
        
        # 선명도 계수 계산
        factor = magnitude * 2.0  # 0 ~ 2.0 범위
        
        # 선명도 적용 (원본과 블러 이미지의 가중 평균)
        sharpened = spec_np + factor * (spec_np - blurred)
        sharpened = np.clip(sharpened, 0, 1)
        
        # 텐서로 다시 변환
        spec_aug = spec.clone()
        spec_aug[0] = torch.from_numpy(sharpened.astype(np.float32))
        
        return spec_aug
    
    def _cutout(self, spec, magnitude):
        """Cutout: 작은 사각형 영역을 여러 개 마스킹"""
        spec_aug = spec.clone()
        
        height, width = spec.shape[1], spec.shape[2]
        num_cutouts = random.randint(1, 4)
        
        size = int(magnitude * min(height, width) * 0.3) + 1
        
        for _ in range(num_cutouts):
            y = random.randint(0, height - size)
            x = random.randint(0, width - size)
            
            spec_aug[0, y:y+size, x:x+size] = 0
        
        return spec_aug

class BirdCLEFDatasetFromNPY(Dataset):
    def __init__(self, df, cfg, spectrograms=None, mode="train"):
        self.df = df
        self.cfg = cfg
        self.mode = mode
        self.spectrograms = spectrograms
        self.audio_data_dir = cfg.audio_data_dir
        
        # 향상된 증강 기법을 위한 객체 생성
        self.augmentations = AudioAugmentations(cfg)
        
        taxonomy_df = pd.read_csv(self.cfg.taxonomy_csv)
        self.species_ids = taxonomy_df['primary_label'].tolist()
        self.num_classes = len(self.species_ids)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.species_ids)}

        if 'filepath' not in self.df.columns:
            self.df['filepath'] = self.cfg.train_datadir + '/' + self.df.filename
        
        if 'samplename' not in self.df.columns:
            self.df['samplename'] = self.df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
        
        if cfg.debug:
            self.df = self.df.sample(min(1000, len(self.df)), random_state=cfg.seed).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        primary_label = row['primary_label']
        filename = row['filename'].split('/')[-1].split('.')[0]
        
        # middle7_ 파일이 존재하는지 확인하고 존재하면 선택 옵션에 포함
        segment_options = ['first7_', 'last7_']
        middle_path = os.path.join(self.audio_data_dir, primary_label, f"middle7_{filename}.npy")
        if os.path.exists(middle_path):
            segment_options.append('middle7_')
        
        # 가능한 옵션 중에서만 랜덤하게 선택
        segment_type = random.choice(segment_options)
        
        # 선택한 세그먼트 타입으로 파일 경로 구성
        npy_path = os.path.join(self.audio_data_dir, primary_label, f"{segment_type}{filename}.npy")
        
        try:
            # npy 파일에서 오디오 데이터 로드
            audio_data = np.load(npy_path)
            
            # 오디오를 멜 스펙트로그램으로 변환
            spec = audio2melspec(audio_data, self.cfg)
            
            # 스펙트로그램 크기 조정
            if spec.shape != self.cfg.TARGET_SHAPE:
                spec = cv2.resize(spec, self.cfg.TARGET_SHAPE, interpolation=cv2.INTER_LINEAR)
                
        except Exception as e:
            # 파일을 찾을 수 없거나 처리 오류가 있는 경우
            if self.mode == "train":
                print(f"Warning: Error processing {npy_path}: {e}")
            spec = np.zeros(self.cfg.TARGET_SHAPE, dtype=np.float32)
        
        spec = torch.tensor(spec, dtype=torch.float32)
        
        # 채널 추가 (수정됨 - 1채널만 사용)
        spec = spec.unsqueeze(0)  # [1, H, W]
        
        # 증강 적용
        if self.mode == "train" and random.random() < self.cfg.aug_prob:
            spec = self.augmentations.apply_all_augmentations(spec)
    
        target = self.encode_label(row['primary_label'])
    
        if 'secondary_labels' in row and row['secondary_labels'] not in [[''], None, np.nan]:
            if isinstance(row['secondary_labels'], str):
                secondary_labels = eval(row['secondary_labels'])
            else:
                secondary_labels = row['secondary_labels']
            
            for label in secondary_labels:
                if label in self.label_to_idx:
                    target[self.label_to_idx[label]] = 1.0
    
        return {
            'melspec': spec,  # [1, H, W]
            'target': torch.tensor(target, dtype=torch.float32),
            'filename': row['filename']
        }
    
    def encode_label(self, label):
        """Encode label to one-hot vector"""
        target = np.zeros(self.num_classes)
        if label in self.label_to_idx:
            target[self.label_to_idx[label]] = 1.0
        return target
    
# Mixup 기능 개선
class MixupAugmentation:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
    
    def apply_mixup(self, x, targets):
        """배치에 Mixup 증강 적용"""
        batch_size = x.size(0)
        
        # Mixup 강도 결정 (베타 분포에서 샘플링)
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # 랜덤 샘플 인덱스 생성
        indices = torch.randperm(batch_size).to(x.device)
        
        # 샘플 혼합
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        return mixed_x, targets, targets[indices], lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup에 맞게 손실 함수 계산"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
def collate_fn(batch):
    """Custom collate function to handle different sized spectrograms"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
        
    result = {key: [] for key in batch[0].keys()}
    
    for item in batch:
        for key, value in item.items():
            result[key].append(value)
    
    for key in result:
        if key == 'target' and isinstance(result[key][0], torch.Tensor):
            result[key] = torch.stack(result[key])
        elif key == 'melspec' and isinstance(result[key][0], torch.Tensor):
            shapes = [t.shape for t in result[key]]
            if len(set(str(s) for s in shapes)) == 1:
                result[key] = torch.stack(result[key])
    
    return result

class BirdCLEFModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        print(f"Number of classes: {len(taxonomy_df)}")
        cfg.num_classes = len(taxonomy_df)

        self.backbone = timm.create_model(
            cfg.model_name,
            pretrained=True,  
            in_chans=cfg.in_channels,
            drop_rate=0.25,    
            drop_path_rate=0.25
        )
        
        backbone_out = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = backbone_out
        self.classifier = nn.Linear(backbone_out, cfg.num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, dict):
            features = features['features']
        if len(features.shape) == 4:
            features = self.pooling(features)
            features = features.view(features.size(0), -1)

        logits = self.classifier(features)
        return logits
    
def get_optimizer(model, cfg):
  
    if cfg.optimizer == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay
        )
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
        
    return optimizer

def get_scheduler(optimizer, cfg):
   
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.T_max,
            eta_min=cfg.min_lr
        )
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=cfg.min_lr,
            verbose=True
        )
    elif cfg.scheduler == 'StepLR':
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.epochs // 3,
            gamma=0.5
        )
    elif cfg.scheduler == 'OneCycleLR':
        scheduler = None  
    else:
        scheduler = None
        
    return scheduler

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', bce_weight=0.5):
        super(BCEFocalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        
    def forward(self, inputs, targets):
        focal_loss = self.focal_loss(inputs, targets)
        bce_loss = self.bce(inputs, targets)
        
        # Average of BCE and FocalLoss
        combined_loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * focal_loss
        
        return combined_loss

def get_criterion(cfg):
    if cfg.criterion == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif cfg.criterion == 'FocalLoss':
        criterion = FocalLoss()
    elif cfg.criterion == 'BCEFocalLoss':
        criterion = BCEFocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    else:
        raise NotImplementedError(f"Criterion {cfg.criterion} not implemented")
        
    return criterion

def train_one_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    
    model.train()
    losses = []
    all_targets = []
    all_outputs = []
    
    pbar = tqdm(enumerate(loader), total=len(loader), desc="Training")
    
    for step, batch in pbar:
    
        if isinstance(batch['melspec'], list):
            batch_outputs = []
            batch_losses = []
            
            for i in range(len(batch['melspec'])):
                inputs = batch['melspec'][i].unsqueeze(0).to(device)
                target = batch['target'][i].unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, target)
                loss.backward()
                
                batch_outputs.append(output.detach().cpu())
                batch_losses.append(loss.item())
            
            optimizer.step()
            outputs = torch.cat(batch_outputs, dim=0).numpy()
            loss = np.mean(batch_losses)
            targets = batch['target'].numpy()
            
        else:
            inputs = batch['melspec'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if isinstance(outputs, tuple):
                outputs, loss = outputs  
            else:
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            
            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
        
        if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
            scheduler.step()
            
        all_outputs.append(outputs)
        all_targets.append(targets)
        losses.append(loss if isinstance(loss, float) else loss.item())
        
        pbar.set_postfix({
            'train_loss': np.mean(losses[-10:]) if losses else 0,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def validate(model, loader, criterion, device):
   
    model.eval()
    losses = []
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            if isinstance(batch['melspec'], list):
                batch_outputs = []
                batch_losses = []
                
                for i in range(len(batch['melspec'])):
                    inputs = batch['melspec'][i].unsqueeze(0).to(device)
                    target = batch['target'][i].unsqueeze(0).to(device)
                    
                    output = model(inputs)
                    loss = criterion(output, target)
                    
                    batch_outputs.append(output.detach().cpu())
                    batch_losses.append(loss.item())
                
                outputs = torch.cat(batch_outputs, dim=0).numpy()
                loss = np.mean(batch_losses)
                targets = batch['target'].numpy()
                
            else:
                inputs = batch['melspec'].to(device)
                targets = batch['target'].to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                outputs = outputs.detach().cpu().numpy()
                targets = targets.detach().cpu().numpy()
            
            all_outputs.append(outputs)
            all_targets.append(targets)
            losses.append(loss if isinstance(loss, float) else loss.item())
    
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    auc = calculate_auc(all_targets, all_outputs)
    avg_loss = np.mean(losses)
    
    return avg_loss, auc

def calculate_auc(targets, outputs):
  
    num_classes = targets.shape[1]
    aucs = []
    
    probs = 1 / (1 + np.exp(-outputs))
    
    for i in range(num_classes):
        
        if np.sum(targets[:, i]) > 0:
            class_auc = roc_auc_score(targets[:, i], probs[:, i])
            aucs.append(class_auc)
    
    return np.mean(aucs) if aucs else 0.0

def run_training(df, cfg):
    """Training function for 10-second audio segments"""
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
    species_ids = taxonomy_df['primary_label'].tolist()
    cfg.num_classes = len(species_ids)
    
    if cfg.debug:
        cfg.update_debug_settings()

    # spectrograms 변수를 None으로 초기화 (10초 데이터를 직접 로드할 것이므로)
    spectrograms = None
    
    # 변경: 단일 npy 파일을 로드하지 않음
    if 'filepath' not in df.columns:
        df['filepath'] = cfg.train_datadir + '/' + df.filename
    if 'samplename' not in df.columns:
        df['samplename'] = df.filename.map(lambda x: x.split('/')[0] + '-' + x.split('/')[-1].split('.')[0])
    
    # 데이터 검증: 10초 데이터 폴더에 실제로 파일이 있는지 확인
    print("Validating 10-second data availability...")
    sample_label = df['primary_label'].iloc[0]
    sample_file = df['filename'].iloc[0].split('/')[-1].split('.')[0]
    sample_path = os.path.join(cfg.audio_data_dir, sample_label, f"first7_{sample_file}.npy")
    
    if os.path.exists(sample_path):
        print(f"10-second data validated: Found {sample_path}")
    else:
        print(f"Warning: 10-second data not found at {sample_path}")
        print("Please check the data path and structure.")
    
    # 나머지 코드는 유지
    skf = StratifiedKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    
    best_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        if fold not in cfg.selected_folds:
            continue
            
        print(f'\n{"="*30} Fold {fold} {"="*30}')
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f'Training set: {len(train_df)} samples')
        print(f'Validation set: {len(val_df)} samples')
        
        train_dataset = BirdCLEFDatasetFromNPY(train_df, cfg, spectrograms=spectrograms, mode='train')
        val_dataset = BirdCLEFDatasetFromNPY(val_df, cfg, spectrograms=spectrograms, mode='valid')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False, 
            num_workers=cfg.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        model = BirdCLEFModel(cfg).to(cfg.device)
        optimizer = get_optimizer(model, cfg)
        criterion = get_criterion(cfg)
        
        if cfg.scheduler == 'OneCycleLR':
            scheduler = lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                steps_per_epoch=len(train_loader),
                epochs=cfg.epochs,
                pct_start=0.1
            )
        else:
            scheduler = get_scheduler(optimizer, cfg)
        
        best_auc = 0
        best_epoch = 0
        
        # 조기 종료를 위한 카운터 추가
        patience = 7  # 7번의 에폭 동안 개선이 없으면 중단
        patience_counter = 0
        
        for epoch in range(cfg.epochs):
            print(f"\nEpoch {epoch+1}/{cfg.epochs}")
            
            train_loss, train_auc = train_one_epoch(
                model, 
                train_loader, 
                optimizer, 
                criterion, 
                cfg.device,
                scheduler if isinstance(scheduler, lr_scheduler.OneCycleLR) else None
            )
            
            val_loss, val_auc = validate(model, val_loader, criterion, cfg.device)

            if scheduler is not None and not isinstance(scheduler, lr_scheduler.OneCycleLR):
                if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch + 1
                patience_counter = 0  # 성능이 개선되었으므로 카운터 초기화
                print(f"New best AUC: {best_auc:.4f} at epoch {best_epoch}")

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'train_auc': train_auc,
                    'cfg': cfg
                }, f"model_fold{fold}.pth")
            else:
                patience_counter += 1  # 성능 개선이 없으면 카운터 증가
                print(f"No improvement in AUC. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs!")
                    break  # 조기 종료
        
        best_scores.append(best_auc)
        print(f"\nBest AUC for fold {fold}: {best_auc:.4f} at epoch {best_epoch}")
        
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("Cross-Validation Results:")
    for fold, score in enumerate(best_scores):
        print(f"Fold {cfg.selected_folds[fold]}: {score:.4f}")
    print(f"Mean AUC: {np.mean(best_scores):.4f}")
    print("="*60)

if __name__ == "__main__":
    cfg = CFG()
    set_seed(cfg.seed)
    print("\nLoading training data...")
    train_df = pd.read_csv(cfg.train_csv)
    taxonomy_df = pd.read_csv(cfg.taxonomy_csv)

    print("\nStarting training...")
    print(f"LOAD_DATA is set to {cfg.LOAD_DATA}")
    if cfg.LOAD_DATA:
        print("Using pre-computed mel spectrograms from NPY file")
    else:
        print("Will generate spectrograms on-the-fly during training")
    
    run_training(train_df, cfg)
    
    print("\nTraining complete!")