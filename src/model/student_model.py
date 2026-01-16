import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import wandb
import logging
import argparse
from scipy.stats import gaussian_kde
import time
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

INPUT_FILE = "/scratch/akeluska/prot_distill_divide/tmvec2_pairs_predictions.parquet"

# WandB configuration
WANDB_CONFIG = {
    "entity": None,
    "project": "tm-distill",
}

# Amino acid vocabulary
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWYXUBZ'
VOCAB_SIZE = len(AMINO_ACIDS) + 2
PAD_TOKEN = 0
UNK_TOKEN = 1
AA_TO_IDX = {aa: i + 2 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX['<PAD>'] = PAD_TOKEN
AA_TO_IDX['<UNK>'] = UNK_TOKEN


# ==============================================================================
# SEQUENCE ENCODING
# ==============================================================================

def encode_sequence(sequence, max_length=None):
    """Encode amino acid sequence to tensor."""
    encoded = []
    for aa in sequence:
        encoded.append(AA_TO_IDX.get(aa, UNK_TOKEN))

    if max_length:
        if len(encoded) > max_length:
            encoded = encoded[:max_length]
        else:
            encoded.extend([PAD_TOKEN] * (max_length - len(encoded)))

    return torch.tensor(encoded, dtype=torch.long)


# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class ProteinSequenceEncoder(nn.Module):
    """BiLSTM encoder with attention pooling."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, output_dim=256, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim // 2, 2,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        mask = (x != PAD_TOKEN).float()
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)

        # Attention pooling
        attention_scores = self.attention(lstm_out).squeeze(-1)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1)

        pooled = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        output = self.output_proj(pooled)

        return output


class TMScorePredictor(nn.Module):
    def __init__(self, combined_dim, dropout=0.1):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.temperature = nn.Parameter(torch.ones(1))
        self.bias_correction = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        logits = self.predictor(x)
        calibrated_logits = logits / torch.clamp(self.temperature, 0.1, 10.0) + self.bias_correction
        output = torch.sigmoid(calibrated_logits).squeeze(-1)
        return output


class StudentModel(nn.Module):
    """Complete model with enhanced architecture."""

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=128, hidden_dim=256,
                 seq_embed_dim=256, dropout=0.1):
        super().__init__()

        self.seq_encoder = ProteinSequenceEncoder(
            vocab_size, embed_dim, hidden_dim, seq_embed_dim, dropout
        )

        combined_dim = seq_embed_dim * 4
        self.tm_predictor = TMScorePredictor(combined_dim, dropout)

    def forward(self, seq_a, seq_b=None):
        repr_a = self.seq_encoder(seq_a)

        if seq_b is not None:
            repr_b = self.seq_encoder(seq_b)

            # 3-way combination
            concat_repr = torch.cat([repr_a, repr_b], dim=1)
            product_repr = repr_a * repr_b
            diff_repr = torch.abs(repr_a - repr_b)

            combined = torch.cat([concat_repr, product_repr, diff_repr], dim=1)
            tm_score_pred = self.tm_predictor(combined)

            return repr_a, repr_b, tm_score_pred
        else:
            return repr_a


# ==============================================================================
# DATASET
# ==============================================================================

class ProteinPairDataset(Dataset):
    """Dataset for protein pairs with TM-scores."""

    def __init__(self, df, max_length=600):
        self.df = df
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq_a = encode_sequence(row['seq_a'], self.max_length)
        seq_b = encode_sequence(row['seq_b'], self.max_length)
        tm_score = torch.tensor(row['tm_score'], dtype=torch.float32)

        return {
            'seq_a': seq_a,
            'seq_b': seq_b,
            'tm_score': tm_score
        }

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.5,
        weight: float = 2.0,
        problem_range_weight: float = 2.5
    ):
        super().__init__()
        self.alpha = alpha
        self.weight = weight
        self.problem_range_weight = problem_range_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        squared_errors = (predictions - targets) ** 2
        abs_errors = torch.abs(predictions - targets)
        focal_weights = torch.pow(abs_errors, self.alpha)

        mask = (targets > 0.6)
        weights = torch.where(
            mask,
            torch.tensor(self.weight, device=targets.device),
            torch.tensor(1.0, device=targets.device)
        )

        problem_mask = (targets >= 0.5) & (targets <= 0.75)
        problem_weights = torch.where(
            problem_mask,
            torch.tensor(self.problem_range_weight, device=targets.device),
            torch.tensor(1.0, device=targets.device)
        )

        total_weights = focal_weights * weights * problem_weights
        weighted_loss = (squared_errors * total_weights).mean()

        info = {
            'focal_loss': weighted_loss.item(),
            'pairs': mask.sum().item(),
            'problem_range_pairs': problem_mask.sum().item(),
            'mean_weight': total_weights.mean().item(),
            'max_weight': total_weights.max().item()
        }
        return weighted_loss, info


class LabelDistributionSmoothingLoss(nn.Module):
    """Label distribution smoothing (LDS) for continuous imbalanced regression."""

    def __init__(
        self,
        train_tm_scores: Optional[np.ndarray] = None,
        kernel_width: float = 0.05,
        reweight_factor: float = 0.05,
        use_precomputed: bool = True
    ):
        super().__init__()
        self.kernel_width = kernel_width
        self.reweight_factor = reweight_factor
        self.use_precomputed = use_precomputed

        self.weight_lookup = None
        self.tm_grid = None
        self.kde = None

        if train_tm_scores is not None:
            self.fit_density(train_tm_scores)

    def fit_density(self, tm_scores: np.ndarray):
        print(f"Computing label density with bandwidth={self.kernel_width}...")
        self.kde = gaussian_kde(tm_scores, bw_method=self.kernel_width)
        print("✓ Label density computed")

        if self.use_precomputed:
            print("Precomputing weight lookup table for fast training...")
            self.tm_grid = np.linspace(0, 1, 1001)
            densities = self.kde(self.tm_grid)
            weights = 1.0 / (densities + self.reweight_factor)
            weights = weights / weights.mean()
            self.weight_lookup = torch.tensor(weights, dtype=torch.float32)
            print(f"✓ Precomputed weights for {len(self.tm_grid)} grid points")

    def get_weights(self, targets: torch.Tensor) -> torch.Tensor:
        if self.kde is None:
            return torch.ones_like(targets)

        if self.use_precomputed and self.weight_lookup is not None:
            if self.weight_lookup.device != targets.device:
                self.weight_lookup = self.weight_lookup.to(targets.device)

            indices = (targets * 1000).long().clamp(0, 1000)
            return self.weight_lookup[indices]

        targets_np = targets.cpu().numpy()
        densities = self.kde(targets_np)
        weights = 1.0 / (densities + self.reweight_factor)
        weights = weights / weights.mean()
        return torch.tensor(weights, device=targets.device, dtype=torch.float32)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        weights = self.get_weights(targets)
        squared_errors = (predictions - targets) ** 2
        weighted_loss = (squared_errors * weights).mean()
        info = {
            'lds_loss': weighted_loss.item(),
            'mean_weight': weights.mean().item(),
            'max_weight': weights.max().item(),
            'min_weight': weights.min().item()
        }
        return weighted_loss, info


class ContrastiveTMScoreLoss(nn.Module):
    """Contrastive loss for protein embeddings based on TM-scores."""

    def __init__(
        self,
        margin: float = 0.2,
        temperature: float = 0.07,
        similarity_threshold: float = 0.6,
        dissimilarity_threshold: float = 0.17
    ):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.dissimilarity_threshold = dissimilarity_threshold

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        tm_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        embeddings_a_norm = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b_norm = F.normalize(embeddings_b, p=2, dim=1)
        cosine_sim = (embeddings_a_norm * embeddings_b_norm).sum(dim=1)

        similar_mask = tm_scores > self.similarity_threshold
        dissimilar_mask = tm_scores < self.dissimilarity_threshold

        similar_loss = torch.tensor(0.0, device=tm_scores.device)
        if similar_mask.sum() > 0:
            similar_loss = (1.0 - cosine_sim[similar_mask]).mean()

        dissimilar_loss = torch.tensor(0.0, device=tm_scores.device)
        if dissimilar_mask.sum() > 0:
            dissimilar_loss = F.relu(cosine_sim[dissimilar_mask] - self.margin).mean()

        total_loss = similar_loss + dissimilar_loss
        info = {
            'contrastive_loss': total_loss.item(),
            'similar_loss': similar_loss.item(),
            'dissimilar_loss': dissimilar_loss.item(),
            'n_similar': similar_mask.sum().item(),
            'n_dissimilar': dissimilar_mask.sum().item(),
            'mean_cosine_sim': cosine_sim.mean().item()
        }
        return total_loss, info


class TMScoreLoss(nn.Module):
    """Combined enhanced loss function incorporating all improvements."""

    def __init__(
        self,
        train_tm_scores: Optional[np.ndarray] = None,
        focal_weight: float = 0.5,
        lds_weight: float = 0.3,
        contrastive_weight: float = 0.1,
        range_penalty_weight: float = 0.1,
        use_focal: bool = True,
        use_lds: bool = True,
        use_contrastive: bool = True
    ):
        super().__init__()
        self.focal_loss = FocalLoss() if use_focal else None
        self.lds_loss = LabelDistributionSmoothingLoss(train_tm_scores) if use_lds else None
        self.contrastive_loss = ContrastiveTMScoreLoss() if use_contrastive else None

        self.focal_weight = focal_weight
        self.lds_weight = lds_weight
        self.contrastive_weight = contrastive_weight
        self.range_penalty_weight = range_penalty_weight

        self.use_focal = use_focal
        self.use_lds = use_lds
        self.use_contrastive = use_contrastive

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        embeddings_a: Optional[torch.Tensor] = None,
        embeddings_b: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        info = {}
        base_mse = F.mse_loss(predictions, targets)
        info['base_mse'] = base_mse.item()

        total_loss = base_mse

        if self.use_focal and self.focal_loss is not None:
            focal_loss, focal_info = self.focal_loss(predictions, targets)
            total_loss = total_loss + self.focal_weight * focal_loss
            info.update({f'focal_{k}': v for k, v in focal_info.items()})
            info['focal_weight_applied'] = self.focal_weight

        if self.use_lds and self.lds_loss is not None:
            lds_loss, lds_info = self.lds_loss(predictions, targets)
            total_loss = total_loss + self.lds_weight * lds_loss
            info.update({f'lds_{k}': v for k, v in lds_info.items()})
            info['lds_weight_applied'] = self.lds_weight

        if self.use_contrastive and self.contrastive_loss is not None:
            if embeddings_a is not None and embeddings_b is not None:
                contrastive_loss, contrastive_info = self.contrastive_loss(
                    embeddings_a, embeddings_b, targets
                )
                total_loss = total_loss + self.contrastive_weight * contrastive_loss
                info.update({f'contrastive_{k}': v for k, v in contrastive_info.items()})
                info['contrastive_weight_applied'] = self.contrastive_weight

        range_penalty = (
            torch.mean(F.relu(predictions - 1.0) ** 2) +
            torch.mean(F.relu(-predictions) ** 2)
        )
        total_loss = total_loss + self.range_penalty_weight * range_penalty
        info['range_penalty'] = range_penalty.item()
        info['range_penalty_weight_applied'] = self.range_penalty_weight
        info['total_loss'] = total_loss.item()
        return total_loss, info

def get_metrics(true_scores, pred_scores):
    """Compute bucket-wise and overall metrics."""
    true_scores = np.array(true_scores)
    pred_scores = np.array(pred_scores)

    metrics = {}

    # Overall metrics
    metrics['overall_r2'] = r2_score(true_scores, pred_scores)
    metrics['overall_mae'] = mean_absolute_error(true_scores, pred_scores)
    metrics['overall_mse'] = np.mean((true_scores - pred_scores) ** 2)
    metrics['bias'] = np.mean(pred_scores - true_scores)

    # Bucket-specific metrics
    buckets = {
        'bucket_1': (0.0, 0.17),
        'bucket_2': (0.17, 0.60),
        'bucket_3': (0.60, 1.0)
    }

    for bucket_name, (min_val, max_val) in buckets.items():
        if bucket_name == 'bucket_3':
            mask = true_scores >= min_val
        else:
            mask = (true_scores >= min_val) & (true_scores < max_val)

        if mask.sum() > 1:
            metrics[f'{bucket_name}_r2'] = r2_score(true_scores[mask], pred_scores[mask])
            metrics[f'{bucket_name}_mae'] = mean_absolute_error(true_scores[mask], pred_scores[mask])
            metrics[f'{bucket_name}_bias'] = np.mean(pred_scores[mask] - true_scores[mask])
            metrics[f'{bucket_name}_count'] = mask.sum()
        else:
            metrics[f'{bucket_name}_r2'] = 0.0
            metrics[f'{bucket_name}_mae'] = 0.0
            metrics[f'{bucket_name}_bias'] = 0.0
            metrics[f'{bucket_name}_count'] = 0

    return metrics


# ==============================================================================
# PLOTTING
# ==============================================================================

def create_scatter_plot(true_scores, pred_scores, epoch, r2, plot_dir, save_name):
    """Create scatter plot of predictions."""
    plt.figure(figsize=(10, 10))

    plt.scatter(true_scores, pred_scores, alpha=0.5, s=8, color='blue')

    # Perfect correlation line
    min_val = 0.0
    max_val = 1.0
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect')

    # Best fit line
    z = np.polyfit(true_scores, pred_scores, 1)
    p = np.poly1d(z)
    plt.plot(true_scores, p(true_scores), 'g-', linewidth=2, alpha=0.8,
             label=f'Fit (slope={z[0]:.3f})')

    # TM-Score Thresholds
    plt.axvline(0.17, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='middle zone')
    plt.axvline(0.60, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

    plt.xlabel('True TM-Score', fontsize=14, fontweight='bold')
    plt.ylabel('Predicted TM-Score', fontsize=14, fontweight='bold')
    plt.title(f'Epoch {epoch+1}: Enhanced Model with middle Zone Focus\nR² = {r2:.6f}',
              fontsize=14, fontweight='bold')

    plt.gca().set_xticks([0.0, 0.17, 0.60, 1.0])
    plt.gca().set_yticks([0.0, 0.17, 0.60, 1.0])
    plt.gca().tick_params(axis='both', which='major', labelsize=12)

    stats_text = f'R² = {r2:.6f}\nCorr = {np.corrcoef(true_scores, pred_scores)[0,1]:.6f}\nSamples = {len(true_scores):,}'
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
             verticalalignment='top', fontsize=11)

    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(plot_dir, f"{save_name}_epoch_{epoch+1:02d}_r2_{r2:.6f}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


# ==============================================================================
# TRAINING
# ==============================================================================

def train_student_model(
    data_path,
    batch_size=64,
    num_epochs=25,
    learning_rate=1e-3,
    device='cuda',
    max_length=600,
    train_split=0.85,
    max_samples=None
):
    config = {
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "max_length": max_length,
    }

    # Initialize WandB
    wandb.init(
        entity=WANDB_CONFIG["entity"],
        project=WANDB_CONFIG["project"],
        config=config,
        reinit=True
    )
    logger.info(f"WandB initialized: {wandb.run.url}")

    # Create output directories
    plot_dir = "student_training_plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Load and split data
    print(f"\nLoading dataset from: {data_path}")
    df = pd.read_parquet(data_path)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
        print(f"Using {len(df):,} samples for testing")

    train_size = int(len(df) * train_split)
    train_df = df[:train_size].copy()
    val_df = df[train_size:].copy()

    print(f"Training: {len(train_df):,}, Validation: {len(val_df):,}")

    # Create datasets
    train_dataset = ProteinPairDataset(train_df, max_length=max_length)
    val_dataset = ProteinPairDataset(val_df, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize model
    model = StudentModel().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Initialize enhanced loss
    print("\nInitializing enhanced loss (full configuration)...")
    
    # Print the columns of train_df for debugging
    print(f"Train DataFrame columns: {train_df.columns.tolist()}")

    train_tm_scores = train_df['tm_score'].values
    criterion = TMScoreLoss(
        train_tm_scores=train_tm_scores,
        focal_weight=0.5,
        lds_weight=0.3,
        contrastive_weight=0.1,
        range_penalty_weight=0.1,
        use_focal=True,
        use_lds=True,
        use_contrastive=True
    )
    print("✓ Enhanced loss initialized")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

    # Training loop
    best_mae = float('inf')
    best_metrics = {}

    print(f"\nStarting training for {num_epochs} epochs...\n")

    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss_info = {
            'total_loss': 0,
            'base_mse': 0,
            'focal_loss': 0,
            'lds_loss': 0,
            'contrastive_loss': 0
        }

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(train_pbar):
            seq_a = batch['seq_a'].to(device)
            seq_b = batch['seq_b'].to(device)
            tm_score_true = batch['tm_score'].to(device)

            optimizer.zero_grad()

            # Forward pass
            repr_a, repr_b, tm_score_pred = model(seq_a, seq_b)

            # Enhanced loss (with all components)
            loss, loss_info = criterion(
                predictions=tm_score_pred,
                targets=tm_score_true,
                embeddings_a=repr_a,
                embeddings_b=repr_b
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Accumulate loss info
            for key in epoch_loss_info:
                if key in loss_info:
                    epoch_loss_info[key] += loss_info[key]

            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'LR': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # Validation
        model.eval()
        all_preds = []
        all_true = []

        with torch.no_grad():
            for batch in val_loader:
                seq_a = batch['seq_a'].to(device)
                seq_b = batch['seq_b'].to(device)
                tm_score_true = batch['tm_score']

                _, _, tm_score_pred = model(seq_a, seq_b)

                all_preds.extend(tm_score_pred.cpu().numpy())
                all_true.extend(tm_score_true.numpy())

        # Compute metrics
        metrics = get_metrics(all_true, all_preds)

        # Create scatter plot
        plot_path = create_scatter_plot(
            all_true, all_preds, epoch, metrics['overall_r2'],
            plot_dir, "enhanced"
        )

        # Save best model
        if metrics['overall_mae'] < best_mae:
            curr_stamp = time.time()
            print(f"Save new best model at timestamp {curr_stamp}")
            best_mae = metrics['overall_mae']
            best_metrics = metrics.copy()
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'metrics': metrics,
                'overall_mae': metrics['overall_mae'],
                'config': config
            }, f"student_best_{curr_stamp}.pt")

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs} Results:")
        print(f"  Overall: R²={metrics['overall_r2']:.6f}, MAE={metrics['overall_mae']:.6f}")
        print(f"  Plot: {os.path.basename(plot_path)}")

        # WandB logging
        log_dict = {
            "epoch": epoch + 1,
            **metrics,
            **{f"loss_{k}": v/len(train_loader) for k, v in epoch_loss_info.items()},
            "calibration_temp": model.tm_predictor.temperature.item(),
            "bias_correction": model.tm_predictor.bias_correction.item()
        }
        wandb.log(log_dict)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETED!")
    print(f"{'='*70}")
    print("\nBest model performance:")
    print(f"  Overall: R²={best_metrics['overall_r2']:.6f}, MAE={best_metrics['overall_mae']:.6f}")
    print("\nModel saved to: student_best.pt")
    print(f"{'='*70}")

    wandb.finish()

    return model, best_metrics

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train enhanced TM-score prediction model')
    parser.add_argument('--data', type=str, default=INPUT_FILE,
                       help='Path to balanced dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=25,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples (for testing)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to train on')


    args = parser.parse_args()

    # Train on the specified data
    model, best_metrics = train_student_model(
        data_path=args.data,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=args.device,
        max_samples=args.max_samples
    )

    print("\n✓ Training complete! Check student_best.pt for the trained model.")


if __name__ == '__main__':
    main()