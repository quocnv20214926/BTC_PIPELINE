"""Feature engineering and CUDA 1D-CNN training helpers for BTCUSDT M15."""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler


@dataclass
class CNNConfig:
    lookback: int = 48
    horizon: int = 4
    label_threshold: float = 0.004
    risk_floor: float = 0.0025
    log_rr_clip: float = 2.0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    batch_size: int = 512
    epochs: int = 15
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    rr_loss_weight: float = 0.30
    class_weight_power: float = 1.0
    dropout: float = 0.25
    seed: int = 42


FEATURE_NAMES = [
    'co_return', 'high_shadow_return', 'low_shadow_return', 'candle_range_return',
    'log_return_1', 'log_return_4', 'log_return_12',
    'ema20_distance', 'ema200_distance_atr',
    'atr14_pct', 'volatility_12', 'volatility_48',
    'volume_log_ratio_1', 'volume_log_ratio_8', 'volume_zscore_48',
    'position_in_range_48', 'rsi14_scaled', 'time_sin', 'time_cos',
]
CLASS_NAMES = np.array(['HOLD', 'LONG', 'SHORT'])


def engineer_features(df):
    """Return a compact, non-forward-looking feature frame."""
    eps = 1e-12
    f = pd.DataFrame(index=df.index)
    body = df.close - df.open
    candle_range = (df.high - df.low).clip(lower=eps)
    f['co_return'] = body / df.open
    f['high_shadow_return'] = (df.high - df[['open', 'close']].max(axis=1)) / df.open
    f['low_shadow_return'] = (df[['open', 'close']].min(axis=1) - df.low) / df.open
    f['candle_range_return'] = candle_range / df.open
    log_close = np.log(df.close)
    for lag in (1, 4, 12):
        f[f'log_return_{lag}'] = log_close.diff(lag)
    ema20 = df.close.ewm(span=20, adjust=False).mean()
    ema200 = df.close.ewm(span=200, adjust=False).mean()
    f['ema20_distance'] = ((df.close - ema20) / ema20).clip(-0.10, 0.10) / 0.10
    previous_close = df.close.shift(1)
    true_range = pd.concat([(df.high-df.low), (df.high-previous_close).abs(), (df.low-previous_close).abs()], axis=1).max(axis=1)
    atr14 = true_range.ewm(alpha=1/14, adjust=False).mean()
    f['ema200_distance_atr'] = ((df.close-ema200) / atr14.clip(lower=eps)).clip(-10, 10) / 10
    f['atr14_pct'] = atr14 / df.close
    f['volatility_12'] = f.log_return_1.rolling(12).std()
    f['volatility_48'] = f.log_return_1.rolling(48).std()
    log_volume = np.log1p(df.volume)
    f['volume_log_ratio_1'] = log_volume - log_volume.shift(1)
    f['volume_log_ratio_8'] = log_volume - log_volume.shift(8)
    prior_mean = log_volume.rolling(48).mean().shift(1)
    prior_std = log_volume.rolling(48).std().shift(1)
    f['volume_zscore_48'] = (log_volume-prior_mean) / prior_std.clip(lower=eps)
    high48, low48 = df.high.rolling(48).max(), df.low.rolling(48).min()
    f['position_in_range_48'] = (df.close-low48) / (high48-low48).clip(lower=eps)
    delta = df.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
    f['rsi14_scaled'] = (100 - 100/(1 + gain/loss.clip(lower=eps))) / 100
    minute = f.index.hour*60 + f.index.minute
    f['time_sin'] = np.sin(2*np.pi*minute/1440)
    f['time_cos'] = np.cos(2*np.pi*minute/1440)
    return f[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)


def make_labels(df, config):
    """Direction and log Risk:R labels from the four future candles."""
    entry = df.open.shift(-1)
    exit_close = df.close.shift(-config.horizon)
    future_high = pd.concat([df.high.shift(-i) for i in range(1, config.horizon+1)], axis=1).max(axis=1)
    future_low = pd.concat([df.low.shift(-i) for i in range(1, config.horizon+1)], axis=1).min(axis=1)
    future_return = (exit_close-entry)/entry
    label = pd.Series(np.select(
        [future_return > config.label_threshold, future_return < -config.label_threshold],
        [1, 2], default=0), index=df.index, dtype='float64')
    label[exit_close.isna()] = np.nan
    long_reward = ((exit_close-entry)/entry).clip(lower=0)
    short_reward = ((entry-exit_close)/entry).clip(lower=0)
    long_risk = ((entry-future_low)/entry).clip(lower=config.risk_floor)
    short_risk = ((future_high-entry)/entry).clip(lower=config.risk_floor)
    rr = pd.Series(np.nan, index=df.index, dtype='float64')
    rr[label.eq(1)] = (long_reward/long_risk)[label.eq(1)]
    rr[label.eq(2)] = (short_reward/short_risk)[label.eq(2)]
    # Log target preserves the conventional ratio while preventing tiny risk
    # denominators from dominating the regression loss.
    log_rr = np.log(rr.clip(lower=np.exp(-config.log_rr_clip))).clip(-config.log_rr_clip, config.log_rr_clip)
    return label, log_rr, future_return


def prepare_data(path, config):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df.open_time_ms, unit='ms', utc=True)
    df = df.sort_values('timestamp').drop_duplicates('timestamp').set_index('timestamp')
    features = engineer_features(df)
    label, log_rr, future_return = make_labels(df, config)
    valid = features.notna().all(axis=1) & label.notna()
    return df.loc[valid], features.loc[valid], label.loc[valid].astype('int64'), log_rr.loc[valid].astype('float32'), future_return.loc[valid]


def split_and_scale(features, labels, log_rr, config):
    values = features.to_numpy(dtype=np.float32)
    ends = np.arange(config.lookback-1, len(features))
    n_train, n_val = int(len(ends)*config.train_ratio), int(len(ends)*config.val_ratio)
    embargo = config.lookback + config.horizon
    splits = {
        'train': ends[:n_train],
        'validation': ends[n_train+embargo:n_train+embargo+n_val],
        'test': ends[n_train+embargo+n_val+embargo:],
    }
    scaler = StandardScaler().fit(values[:splits['train'][-1]+1])
    scaled = np.clip(scaler.transform(values), -10, 10).astype(np.float32)
    return scaled, labels.to_numpy(), log_rr.to_numpy(), splits, scaler


def torch_components(config, scaled, labels, log_rr, splits):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required. Select the project kernel with CUDA-enabled PyTorch.')
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    class SequenceDataset(Dataset):
        def __init__(self, ends): self.ends = np.asarray(ends)
        def __len__(self): return len(self.ends)
        def __getitem__(self, index):
            end = self.ends[index]
            # Conv1d expects channels/features before sequence length.
            x = scaled[end-config.lookback+1:end+1].T.copy()
            rr = log_rr[end] if np.isfinite(log_rr[end]) else 0.0
            mask = float(labels[end] in (1, 2) and np.isfinite(log_rr[end]))
            return torch.from_numpy(x), torch.tensor(labels[end]), torch.tensor(rr), torch.tensor(mask)

    class MultiTaskCNN(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.GELU(),
                nn.MaxPool1d(2), nn.Dropout(config.dropout),
                nn.Conv1d(64, 96, kernel_size=3, padding=1), nn.BatchNorm1d(96), nn.GELU(),
            )
            self.pool_avg, self.pool_max = nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
            self.shared = nn.Sequential(nn.Linear(192, 64), nn.GELU(), nn.Dropout(config.dropout))
            self.class_head = nn.Linear(64, 3)
            self.rr_head = nn.Linear(64, 1)
        def forward(self, x):
            z = self.encoder(x)
            z = torch.cat([self.pool_avg(z).squeeze(-1), self.pool_max(z).squeeze(-1)], dim=1)
            z = self.shared(z)
            return self.class_head(z), self.rr_head(z).squeeze(1)

    loaders = {}
    for name, ids in splits.items():
        loaders[name] = DataLoader(SequenceDataset(ids), batch_size=config.batch_size,
                                   shuffle=name == 'train', num_workers=0, pin_memory=True)
    return torch, nn, MultiTaskCNN(scaled.shape[1]), loaders


def evaluate(model, loader, torch, device):
    model.eval(); ys=[]; preds=[]; probabilities=[]; rr_true=[]; rr_pred=[]
    with torch.inference_mode():
        for x, y, rr, mask in loader:
            x = x.to(device, non_blocking=True)
            logits, predicted_rr = model(x)
            probability = logits.softmax(1).cpu().numpy()
            ys.append(y.numpy()); preds.append(probability.argmax(1)); probabilities.append(probability)
            trade = mask.bool()
            if trade.any():
                rr_true.append(rr[trade].numpy()); rr_pred.append(predicted_rr.cpu()[trade].numpy())
    y = np.concatenate(ys); pred = np.concatenate(preds); probability = np.concatenate(probabilities)
    rt, rp = np.concatenate(rr_true), np.concatenate(rr_pred)
    median = np.median(rt)
    confidence_policy = {}
    confidence = probability.max(axis=1)
    for threshold in (0.40, 0.50, 0.60, 0.70):
        selected = (pred != 0) & (confidence >= threshold)
        correct = (pred[selected] == y[selected]) if selected.any() else np.array([])
        confidence_policy[str(threshold)] = {
            'signals': int(selected.sum()), 'coverage': float(selected.mean()),
            'precision': float(correct.mean()) if len(correct) else None,
        }
    return {
        'macro_f1': f1_score(y, pred, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'accuracy': float((y == pred).mean()),
        'majority_macro_f1': f1_score(y, np.full_like(y, np.bincount(y).argmax()), average='macro'),
        'log_rr_mae': mean_absolute_error(rt, rp),
        'log_rr_median_baseline_mae': mean_absolute_error(rt, np.full_like(rt, median)),
        'confusion_matrix': confusion_matrix(y, pred).tolist(),
        'classification_report': classification_report(y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        'confidence_policy': confidence_policy,
        'samples': len(y),
    }


def train_cuda(path, output_dir, config=CNNConfig()):
    _, features, labels, log_rr, _ = prepare_data(path, config)
    scaled, y, rr, splits, scaler = split_and_scale(features, labels, log_rr, config)
    torch, nn, model, loaders = torch_components(config, scaled, y, rr, splits)
    device = torch.device('cuda')
    model = model.to(device)
    counts = np.bincount(y[splits['train']], minlength=3)
    raw_weights = len(splits['train'])/(3*counts)
    # power < 1 softens class balancing and usually trades recall for cleaner
    # LONG/SHORT signals; power=0 disables weighting, power=1 is full balance.
    weights = torch.tensor(raw_weights ** config.class_weight_power, dtype=torch.float32, device=device)
    class_loss, rr_loss = nn.CrossEntropyLoss(weight=weights), nn.SmoothL1Loss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=1)
    amp = torch.amp.GradScaler('cuda')
    best_score, best_state, stale, history = -np.inf, None, 0, []
    for epoch in range(config.epochs):
        model.train(); total=0.0
        for x, label, target_rr, mask in loaders['train']:
            x, label = x.to(device, non_blocking=True), label.to(device, non_blocking=True)
            target_rr, mask = target_rr.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                logits, predicted_rr = model(x)
                loss_c = class_loss(logits, label)
                per_item = rr_loss(predicted_rr, target_rr)
                loss_r = (per_item*mask).sum()/mask.sum().clamp_min(1)
                loss = loss_c + config.rr_loss_weight*loss_r
            amp.scale(loss).backward()
            amp.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp.step(optimizer); amp.update(); total += loss.item()*len(label)
        metrics = evaluate(model, loaders['validation'], torch, device)
        scheduler.step(metrics['macro_f1'])
        history.append({'epoch': epoch+1, 'train_loss': total/len(splits['train']),
                        'val_macro_f1': metrics['macro_f1'], 'val_balanced_accuracy': metrics['balanced_accuracy'],
                        'val_log_rr_mae': metrics['log_rr_mae']})
        print(history[-1])
        if metrics['macro_f1'] > best_score + 1e-4:
            best_score, best_state, stale = metrics['macro_f1'], copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= config.patience: break
    model.load_state_dict(best_state)
    validation = evaluate(model, loaders['validation'], torch, device)
    test = evaluate(model, loaders['test'], torch, device)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'n_features': len(FEATURE_NAMES), 'config': asdict(config)}, output_dir/'cnn_1d_cuda.pt')
    joblib.dump(scaler, output_dir/'feature_scaler.joblib')
    metadata = {'device': str(device), 'gpu': torch.cuda.get_device_name(0), 'features': FEATURE_NAMES,
                'input_shape': [config.lookback, len(FEATURE_NAMES)], 'config': asdict(config),
                'class_distribution': {CLASS_NAMES[i]: int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
                'history': history, 'validation': validation, 'test': test}
    (output_dir/'metrics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return model, scaler, metadata


def evaluate_checkpoint(path, checkpoint_path, config=CNNConfig()):
    """Evaluate a saved checkpoint with the current reporting policies."""
    _, features, labels, log_rr, _ = prepare_data(path, config)
    scaled, y, rr, splits, _ = split_and_scale(features, labels, log_rr, config)
    torch, _, model, loaders = torch_components(config, scaled, y, rr, splits)
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    return {name: evaluate(model, loaders[name], torch, torch.device('cuda'))
            for name in ('validation', 'test')}
