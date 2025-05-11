"""
BirdCLEF+ 2025 대회를 위한 학습 스크립트
로컬 환경에서 실행 가능하도록 정리됨
"""
import os
import gc
import time
import copy
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import rcParams
from tqdm import tqdm
from logging import INFO, FileHandler, Formatter, StreamHandler, getLogger

import torch
import torchaudio
import torchvision
from torch.optim import AdamW
from torch.utils.data.sampler import WeightedRandomSampler
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.autograd import Variable

import timm
import albumentations as A
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

warnings.filterwarnings("ignore")


# ========== 설정 ==========
def get_config():
    config = {
        # Experiment settings
        "exp_name": "exp_b0_fast_mel128_fft2048_4s",
        "backbone": "efficientnet_b0",  # Lighter backbone
        "seed": 42,
        "batch_size": 32,
        "num_workers": 4,
        
        # Training settings
        "n_epochs": 10,
        "warmup_epo": 5,
        "lr_max": 1e-5,
        "lr_min": 1e-7,
        "weight_decay": 1e-6,
        "use_amp": True,
        "max_grad_norm": 10,
        "early_stopping": 7,
        
        # Image settings
        "image_size": 224,  # Slightly reduced to save memory
        
        # Audio settings - optimized
        "mel_spec_params": {
            "sample_rate": 32000,
            "n_mels": 128,
            "f_min": 50,
            "f_max": 10000,
            "n_fft": 2048,           # Reduced from 4096
            "hop_length": 256,
            "normalized": True,
            "center": True,
            "pad_mode": "constant",
            "norm": "slaney",
            "onesided": True,
            "mel_scale": "slaney"
        },
        "top_db": 80,
        "train_period": 4,  # Reduced from 5s
        "val_period": 4,
        "secondary_coef": 1.0,
        
        # Cross-validation settings
        "n_fold": 5,
        "fold": 0,  # Use fold 0 only if debugging
        
        # Data paths
        "data_dir": "C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025",
        "train_csv": "C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/train.csv",
        "train_audio_dir": "C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/train_audio",
        "sample_submission": "C:/Users/A.SREE SAI SAMPREETH/Downloads/birdclef-2025/sample_submission.csv",
        
        # Output path
        "output_folder": "./outputs",
    }

    # Derived config
    config["cosine_epo"] = config["n_epochs"] - config["warmup_epo"]
    config["train_duration"] = config["train_period"] * config["mel_spec_params"]["sample_rate"]
    config["val_duration"] = config["val_period"] * config["mel_spec_params"]["sample_rate"]
    
    return config



# ========== 유틸리티 함수 ==========
def set_seed(seed=42):
    """reproducibility를 위한 시드 설정"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger(log_file='train.log'):
    """로깅 설정"""
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


class AverageMeter(object):
    """평균 및 현재 값 계산 및 저장"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# ========== 평가 지표 함수 ==========
def calculate_competition_metrics(y_true, y_pred, target_columns=None):
    """ROC-AUC와 mAP 지표 계산"""
    metrics_dict = {}
    
    # Calculate ROC-AUC for each class, skipping classes with no positive examples
    class_roc_aucs = []
    class_maps = []
    
    for i in range(y_true.shape[1]):
        # Skip if there are no positive examples or all examples are positive
        if np.sum(y_true[:, i]) == 0 or np.sum(y_true[:, i]) == len(y_true[:, i]):
            continue
        
        try:
            roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            class_roc_aucs.append(roc_auc)
            
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            class_maps.append(ap)
            
            # 개별 클래스 ROC-AUC 저장 (선택적)
            if target_columns:
                metrics_dict[f'class_{target_columns[i]}_roc_auc'] = roc_auc
                
        except Exception as e:
            print(f"Error calculating metrics for class {i}: {e}")
    
    if len(class_roc_aucs) > 0:
        metrics_dict['macro_roc_auc'] = np.mean(class_roc_aucs)
        # 리더보드에서 사용되는 약어
        metrics_dict['ROC'] = np.mean(class_roc_aucs)
    else:
        metrics_dict['macro_roc_auc'] = 0.0
        metrics_dict['ROC'] = 0.0
    
    if len(class_maps) > 0:
        metrics_dict['macro_map'] = np.mean(class_maps)
        metrics_dict['mAP'] = np.mean(class_maps)
    else:
        metrics_dict['macro_map'] = 0.0
        metrics_dict['mAP'] = 0.0
    
    return metrics_dict


def calculate_competition_metrics(y_true, y_pred, target_columns=None):
    """ROC-AUC와 mAP 지표 계산"""
    metrics_dict = {}
    
    # Calculate ROC-AUC for each class, skipping classes with no positive examples
    class_roc_aucs = []
    class_maps = []
    
    for i in range(y_true.shape[1]):
        # 레이블이 모두 같은 경우(모두 0 또는 모두 1) 건너뛰기
        if np.sum(y_true[:, i]) == 0 or np.sum(y_true[:, i]) == len(y_true[:, i]):
            continue
        
        try:
            # y_true를 이진 형식으로 변환 (0.5 이상이면 1, 미만이면 0)
            y_true_binary = (y_true[:, i] >= 0.5).astype(int)
            
            # 이진 분류 확인
            if len(np.unique(y_true_binary)) != 2:
                print(f"Warning: Class {i} does not have binary labels")
                continue
                
            roc_auc = roc_auc_score(y_true_binary, y_pred[:, i])
            class_roc_aucs.append(roc_auc)
            
            ap = average_precision_score(y_true_binary, y_pred[:, i])
            class_maps.append(ap)
            
        except Exception as e:
            print(f"Error calculating metrics for class {i}: {e}")
    
    if len(class_roc_aucs) > 0:
        metrics_dict['macro_roc_auc'] = np.mean(class_roc_aucs)
        metrics_dict['ROC'] = np.mean(class_roc_aucs)
    else:
        metrics_dict['macro_roc_auc'] = 0.0
        metrics_dict['ROC'] = 0.0
    
    if len(class_maps) > 0:
        metrics_dict['macro_map'] = np.mean(class_maps)
        metrics_dict['mAP'] = np.mean(class_maps)
    else:
        metrics_dict['macro_map'] = 0.0
        metrics_dict['mAP'] = 0.0
    
    return metrics_dict


def metrics_to_string(metrics_dict, prefix=""):
    """지표 딕셔너리를 문자열로 변환"""
    metrics_str = ""
    
    if prefix:
        metrics_str += f"{prefix} "
    
    # 새로운 지표들을 포함하여 출력
    priority_metrics = ['mse', 'correlation', 'cross_entropy', 'ROC', 'macro_roc_auc', 'macro_map', 'mAP']
    
    for metric_name in priority_metrics:
        if metric_name in metrics_dict:
            metrics_str += f"{metric_name}: {metrics_dict[metric_name]:.4f} "
    
    # 기타 지표들 (클래스별 지표 제외)
    for metric_name, metric_value in metrics_dict.items():
        if metric_name not in priority_metrics and not metric_name.startswith('class_'):
            metrics_str += f"{metric_name}: {metric_value:.4f} "
    
    return metrics_str.strip()


# ========== 스케줄러 클래스 ==========
class GradualWarmupScheduler(_LRScheduler):
    """
    학습률을 점진적으로 웜업(증가)시키는 스케줄러
    'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'에서 제안됨
    """
    
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
    
    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)
    
    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


# GradualWarmupScheduler 버그 수정 버전
class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


# ========== 손실 함수 ==========
class FocalLossBCE(torch.nn.Module):
    def __init__(
            self,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: str = "mean",
            bce_weight: float = 1.0,
            focal_weight: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):
        focall_loss = torchvision.ops.focal_loss.sigmoid_focal_loss(
            inputs=logits,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        bce_loss = self.bce(logits, targets)
        return self.bce_weight * bce_loss + self.focal_weight * focall_loss


# ========== 오디오 처리 함수 ==========
def read_wav(path, sample_rate):
    """오디오 파일 읽기 및 리샘플링"""
    try:
        wav, org_sr = torchaudio.load(path, normalize=True)
        wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=sample_rate)
        return wav
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        # 오류 시 빈 텐서 반환
        return torch.zeros(1, sample_rate * 5)


def crop_start_wav(wav, duration_):
    """오디오 시작부터 자르기"""
    while wav.size(-1) < duration_:
        wav = torch.cat([wav, wav], dim=1)
    wav = wav[:, :duration_]
    return wav


def crop_random_wav(wav, duration_):
    """랜덤 위치에서 5초 선택"""
    while wav.size(-1) < duration_:
        wav = torch.cat([wav, wav], dim=1)
    
    # 랜덤 시작 위치 선택
    if wav.size(-1) > duration_:
        max_start = wav.size(-1) - duration_
        start = random.randint(0, max_start)
        wav = wav[:, start:start+duration_]
    else:
        wav = wav[:, :duration_]
    
    return wav


def filter_human_sound(wav, sample_rate, threshold=0.05):
    """인간 목소리 주파수 대역(300-3000Hz)의 에너지를 감소시킴"""
    try:
        # 푸리에 변환
        fft = torch.fft.rfft(wav)
        freq = torch.fft.rfftfreq(wav.shape[-1], 1/sample_rate)
        
        # 인간 목소리 대역 마스크 생성 (대략 300Hz~3000Hz)
        human_voice_mask = (freq > 300) & (freq < 3000)
        
        # 해당 대역 감쇠
        fft[:, human_voice_mask] *= (1 - threshold)
        
        # 역변환
        filtered_wav = torch.fft.irfft(fft, n=wav.shape[-1])
        return filtered_wav
    except Exception as e:
        print(f"필터링 오류: {e}")
        return wav  # 오류 시 원본 반환


def normalize_melspec(X, eps=1e-6):
    """멜스펙트로그램 정규화"""
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V


def mixup(data, targets, alpha=0.5):
    """Mixup 데이터 증강"""
    indices = torch.randperm(data.size(0))
    data2 = data[indices]
    targets2 = targets[indices]

    lam = torch.FloatTensor([np.random.beta(alpha, alpha)])
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)

    return data, targets


# ========== 데이터셋 클래스 ==========
class BirdDataset(torch.utils.data.Dataset):
    def __init__(self, df, config, transform=None, add_secondary_labels=True, mode='inference'):
        self.df = df
        self.bird2id = config['bird2id']
        self.num_classes = config['num_classes']
        self.secondary_coef = config['secondary_coef']
        self.add_secondary_labels = add_secondary_labels
        self.train_duration = config['train_duration']
        self.mel_spec_params = config['mel_spec_params']
        self.top_db = config['top_db']
        self.mode = mode  # mode 속성 추가
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**self.mel_spec_params)
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=self.top_db)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def prepare_target(self, primary_label, secondary_labels):
        secondary_labels = eval(secondary_labels)
        target = np.zeros(self.num_classes, dtype=np.float32)
        if primary_label != 'nocall':
            primary_label = self.bird2id[primary_label]
            target[primary_label] = 1.0
            if self.add_secondary_labels:
                for s in secondary_labels:
                    if s != "" and s in self.bird2id.keys():
                        target[self.bird2id[s]] = self.secondary_coef
        target = torch.from_numpy(target).float()
        return target

    def prepare_spec(self, path):
        wav = read_wav(path, self.mel_spec_params['sample_rate'])
        wav = crop_random_wav(wav, self.train_duration)  # 랜덤 위치에서 5초 선택
        wav = filter_human_sound(wav, self.mel_spec_params['sample_rate'])  # 인간 목소리 필터링
        mel_spectrogram = normalize_melspec(self.db_transform(self.mel_transform(wav)))
        
        # Time and Freq Masking 추가 (항상 적용하거나 transform이 있는 경우에만 적용)
        # self.mode 대신 self.transform 확인으로 변경
        if self.transform is not None:  # training mode로 가정
            # 시간 마스킹 (Time Masking)
            time_mask_param = int(mel_spectrogram.shape[2] * 0.1)  # 10% 마스킹
            if time_mask_param > 0:
                n_time_masks = np.random.randint(1, 3)  # 1~2개 마스크
                for i in range(n_time_masks):
                    t = np.random.randint(0, time_mask_param)
                    t0 = np.random.randint(0, mel_spectrogram.shape[2] - t)
                    mel_spectrogram[:, :, t0:t0+t] = 0
            
            # 주파수 마스킹 (Frequency Masking)
            freq_mask_param = int(mel_spectrogram.shape[1] * 0.1)  # 10% 마스킹
            if freq_mask_param > 0:
                n_freq_masks = np.random.randint(1, 3)  # 1~2개 마스크
                for i in range(n_freq_masks):
                    f = np.random.randint(0, freq_mask_param)
                    f0 = np.random.randint(0, mel_spectrogram.shape[1] - f)
                    mel_spectrogram[:, f0:f0+f, :] = 0
        
        mel_spectrogram = mel_spectrogram * 255
        mel_spectrogram = mel_spectrogram.expand(3, -1, -1).permute(1, 2, 0).numpy()
        return mel_spectrogram

    def __getitem__(self, idx):
        path = self.df["path"].iloc[idx]
        primary_label = self.df["primary_label"].iloc[idx]
        secondary_labels = self.df["secondary_labels"].iloc[idx]
        rating = self.df["rating"].iloc[idx]

        spec = self.prepare_spec(path)
        target = self.prepare_target(primary_label, secondary_labels)

        if self.transform is not None:
            res = self.transform(image=spec)
            spec = res['image'].astype(np.float32)
        else:
            spec = spec.astype(np.float32)

        spec = spec.transpose(2, 0, 1)

        return {"spec": spec, "target": target, 'rating': rating}


# ========== 모델 클래스 ==========
class GeM(torch.nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        bs, ch, h, w = x.shape
        x = torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(
            1.0 / self.p)
        x = x.view(bs, ch)
        return x


class CNN(torch.nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True):
        super().__init__()

        # 출력할 중간층 인덱스 확장
        out_indices = (2, 3, 4)  # 더 많은 중간층 포함
        self.backbone = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=3,
            num_classes=num_classes,
            out_indices=out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")

        self.global_pools = torch.nn.ModuleList([GeM() for _ in out_indices])
        self.mid_features = np.sum(feature_dims)
        
        # 드롭아웃 추가
        self.dropout = torch.nn.Dropout(0.2)
        self.neck = torch.nn.BatchNorm1d(self.mid_features)
        self.head = torch.nn.Linear(self.mid_features, num_classes)

    def forward(self, x):
        ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        x = self.neck(h)
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.head(x)
        return x


# ========== 학습 및 검증 함수 ==========
# 기존 함수 수정: 실제 적용
def train_one_epoch(model, loader, optimizer, criterion, device, config, scaler=None):
    """한 에폭 학습"""
    model.train()
    losses = AverageMeter()
    gt = []
    preds = []
    bar = tqdm(loader, total=len(loader))
    for batch in bar:
        optimizer.zero_grad()
        spec = batch['spec']
        target = batch['target']

        spec, target = mixup(spec, target, 1.0)

        spec = spec.to(device)
        target = target.to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(spec)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(spec)
            loss = criterion(logits, target)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['max_grad_norm'])
            optimizer.step()

        losses.update(loss.item(), batch["spec"].size(0))
        bar.set_postfix(
            loss=losses.avg,
            grad=grad_norm.item(),
            lr=optimizer.param_groups[0]["lr"]
        )
        gt.append(target.cpu().detach().numpy())
        preds.append(logits.sigmoid().cpu().detach().numpy())
    gt = np.concatenate(gt)
    preds = np.concatenate(preds)
    
    # 새로운 함수 호출로 변경 (calculate_competition_metrics_no_map 대신)
    scores = calculate_train_metrics(gt, preds)

    return scores, losses.avg

def calculate_competition_metrics_no_map(y_true, y_pred, target_columns=None):
    """학습 중에는 단순 손실만 모니터링"""
    return {'ROC': 0.0, 'macro_roc_auc': 0.0}

def calculate_train_metrics(y_true, y_pred):
    """Mixup과 호환되는 지표 계산"""
    metrics_dict = {}
    
    # 1. 평균 제곱 오차 (MSE) - 연속값에 적합
    mse = np.mean((y_true - y_pred) ** 2)
    metrics_dict['mse'] = mse
    
    # 2. 교차 엔트로피 손실 - 확률값에 적합
    # 수치적 안정성을 위한 클리핑
    y_pred_clipped = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    cross_entropy = -np.mean(
        y_true * np.log(y_pred_clipped) + 
        (1 - y_true) * np.log(1.0 - y_pred_clipped)
    )
    metrics_dict['cross_entropy'] = cross_entropy
    
    # 3. 상관계수 (Correlation) - 예측과 실제값의 관계
    # 클래스별 상관계수의 평균
    correlations = []
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
            corr = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    if correlations:
        metrics_dict['correlation'] = np.mean(correlations)
    else:
        metrics_dict['correlation'] = 0.0
    
    # 호환성을 위해 원래 메트릭 이름도 유지
    metrics_dict['ROC'] = metrics_dict['correlation']  # 상관계수를 ROC 대체값으로 사용
    metrics_dict['macro_roc_auc'] = metrics_dict['correlation']
    
    return metrics_dict

def valid_one_epoch(model, loader, criterion, device, config):
    """한 에폭 검증"""
    model.eval()
    losses = AverageMeter()
    bar = tqdm(loader, total=len(loader))
    gt = []
    preds = []

    with torch.no_grad():
        for batch in bar:
            spec = batch['spec'].to(device)
            target = batch['target'].to(device)

            logits = model(spec)
            loss = criterion(logits, target)

            losses.update(loss.item(), batch["spec"].size(0))

            gt.append(target.cpu().detach().numpy())
            preds.append(logits.sigmoid().cpu().detach().numpy())

            bar.set_postfix(loss=losses.avg)

    gt = np.concatenate(gt)
    preds = np.concatenate(preds)
    scores = calculate_competition_metrics(gt, preds, config.get('target_columns'))
    return scores, losses.avg


# ========== 메인 훈련 함수 ==========
def train_fold(config, fold=None):
    """특정 폴드에 대한 모델 훈련"""
    if fold is None:
        fold = config['fold']
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.join(config['output_folder'], config['exp_name']), exist_ok=True)
    
    # 로거 초기화
    logger = init_logger(log_file=os.path.join(config['output_folder'], config['exp_name'], f"{fold}.log"))

    logger.info("=" * 90)
    logger.info(f"Fold {fold} Training")
    logger.info("=" * 90)
    
    # 데이터 준비
    df = config['df']
    trn_df = df[df['fold'] != fold].reset_index(drop=True)
    val_df = df[df['fold'] == fold].reset_index(drop=True)
    
    logger.info(f"Train shape: {trn_df.shape}")
    logger.info(f"Valid shape: {val_df.shape}")
    logger.info(f"Train primary_label value counts:\n{trn_df['primary_label'].value_counts().head(10)}")
    logger.info(f"Valid primary_label value counts:\n{val_df['primary_label'].value_counts().head(10)}")

    # 데이터 변환 설정
    # transforms_train에 RandAugment 추가 (Albumentations에는 직접적인 RandAugment가 없어서 유사한 기능 구현)
    transforms_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Resize(config['image_size'], config['image_size']),
        # RandomErazing과 유사한 CoarseDropout
        A.CoarseDropout(max_height=int(config['image_size'] * 0.375), max_width=int(config['image_size'] * 0.375), max_holes=1, p=0.7),
        # RandAugment 대신 다양한 증강을 랜덤하게 적용
        A.OneOf([
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
            A.ElasticTransform(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Perspective(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.CLAHE(p=0.5),
            A.ColorJitter(p=0.5),
        ], p=0.5),
        # GaussNoise 추가 (대체 변환)
        A.GaussNoise(p=0.2),
        A.Normalize()
    ])

    transforms_val = A.Compose([
        A.Resize(config['image_size'], config['image_size']),
        A.Normalize()
    ])

    # 데이터셋 및 데이터로더 생성
    trn_dataset = BirdDataset(df=trn_df, config=config, transform=transforms_train, add_secondary_labels=True, mode='train')
    val_dataset = BirdDataset(df=val_df, config=config, transform=transforms_val, add_secondary_labels=True, mode='val')

    train_loader = torch.utils.data.DataLoader(
        trn_dataset, 
        shuffle=True, 
        batch_size=config['batch_size'], 
        drop_last=True, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        shuffle=False, 
        batch_size=config['batch_size'], 
        drop_last=False, 
        num_workers=config['num_workers'], 
        pin_memory=True
    )

    # 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 초기화
    model = CNN(
        backbone=config['backbone'], 
        num_classes=config['num_classes'], 
        pretrained=True
    ).to(device)
    
    # 옵티마이저 및 스케줄러 설정
    optimizer = AdamW(model.parameters(), lr=config['lr_max'], weight_decay=config['weight_decay'])
    scheduler_cosine = CosineAnnealingLR(optimizer, config['cosine_epo'])
    scheduler_warmup = GradualWarmupSchedulerV2(
        optimizer, 
        multiplier=10, 
        total_epoch=config['warmup_epo'], 
        after_scheduler=scheduler_cosine
    )
    
    # 손실 함수 초기화
    criterion = FocalLossBCE()
    
    # 그라디언트 스케일러 설정 (혼합 정밀도 학습용)
    scaler = GradScaler() if config['use_amp'] else None
    
    # 조기 종료 관련 변수 초기화
    patience = config['early_stopping']
    best_score = 0.0
    n_patience = 0

    # 학습 반복
    for epoch in range(1, config['n_epochs'] + 1):
        logger.info(f"\n{time.ctime()} Epoch: {epoch}")

        # 스케줄러 스텝
        scheduler_warmup.step(epoch-1)

        # 훈련
        train_scores, train_losses_avg = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config, scaler
        )
        train_scores_str = metrics_to_string(train_scores, "Train")
        train_info = f"Epoch {epoch} - Train loss: {train_losses_avg:.4f}, {train_scores_str}"
        logger.info(train_info)

        # 검증
        val_scores, val_losses_avg = valid_one_epoch(
            model, val_loader, criterion, device, config
        )
        val_scores_str = metrics_to_string(val_scores, "Valid")
        val_info = f"Epoch {epoch} - Valid loss: {val_losses_avg:.4f}, {val_scores_str}"
        logger.info(val_info)

        # 모델 저장 여부 결정
        val_score = val_scores["ROC"]
        is_better = val_score > best_score
        best_score = max(val_score, best_score)

        if is_better:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler_warmup.state_dict() if hasattr(scheduler_warmup, 'state_dict') else None,
            }
            logger.info(f"Epoch {epoch} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                state,
                os.path.join(config['output_folder'], config['exp_name'], f"{fold}.bin")
            )
            n_patience = 0
        else:
            n_patience += 1
            logger.info(f"Valid score didn't improve last {n_patience} epochs.")

        # 조기 종료 체크
        if n_patience >= patience:
            logger.info("Early stopping triggered. Training End.")
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler_warmup.state_dict() if hasattr(scheduler_warmup, 'state_dict') else None,
            }
            torch.save(
                state,
                os.path.join(config['output_folder'], config['exp_name'], f"final_{fold}.bin")
            )
            break

    # 메모리 정리
    del model, optimizer, scheduler_warmup, scheduler_cosine
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_score


# ========== 메인 스크립트 ==========
def main():
    """메인 함수"""
    # 설정 로드
    config = get_config()
    
    # 출력 폴더 생성
    os.makedirs(config['output_folder'], exist_ok=True)
    os.makedirs(os.path.join(config['output_folder'], config['exp_name']), exist_ok=True)
    
    # 시드 설정
    set_seed(config['seed'])
    
    # 데이터 로드
    df = pd.read_csv(config['train_csv'])
    df["path"] = df["filename"].apply(lambda x: os.path.join(config['train_audio_dir'], x))
    df["rating"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
    
    # K-Fold 분할
    skf = StratifiedKFold(n_splits=config['n_fold'], random_state=config['seed'], shuffle=True)
    df['fold'] = -1
    for ifold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df["primary_label"].values)):
        df.loc[val_idx, 'fold'] = ifold
    
    # 클래스 정보 로드
    sub = pd.read_csv(config['sample_submission'])
    target_columns = sub.columns.tolist()[1:]
    num_classes = len(target_columns)
    bird2id = {b: i for i, b in enumerate(target_columns)}
    
    # 설정에 추가 정보 저장
    config['df'] = df
    config['target_columns'] = target_columns
    config['num_classes'] = num_classes
    config['bird2id'] = bird2id
    
    print(f"데이터셋 크기: {len(df)}")
    print(f"타겟 클래스 수: {num_classes}")
    
    # 학습률 시각화
    rcParams['figure.figsize'] = 20, 2
    optimizer = AdamW([torch.nn.Parameter(torch.zeros(1))], lr=config['lr_max'], weight_decay=config['weight_decay'])
    scheduler_cosine = CosineAnnealingLR(optimizer, config['cosine_epo'])
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=config['warmup_epo'], after_scheduler=scheduler_cosine)

    lrs = []
    for epoch in range(1, config['n_epochs']):
        scheduler_warmup.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    plt.plot(range(len(lrs)), lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.savefig(os.path.join(config['output_folder'], config['exp_name'], 'lr_schedule.png'))
    plt.close()
    
    # 특정 폴드 학습
    # fold = config['fold']
    # print(f"===== Training Fold {fold} =====")
    # best_score = train_fold(config, fold)
    # print(f"Fold {fold} best score: {best_score:.4f}")
    
    
    # 모든 폴드 학습 (선택적)
    fold_scores = []
    for fold in range(config['n_fold']):
        print(f"===== Training Fold {fold} =====")
        best_score = train_fold(config, fold)
        fold_scores.append(best_score)
        print(f"Fold {fold} best score: {best_score:.4f}")
    
    # 전체 결과 출력
    print("\n===== Cross-Validation Results =====")
    for fold, score in enumerate(fold_scores):
        print(f"Fold {fold}: {score:.4f}")
    print(f"Mean CV Score: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    


if __name__ == "__main__":
    main()