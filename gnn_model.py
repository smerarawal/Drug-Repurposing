"""
gnn_model.py — Graph Neural Network for binding affinity prediction

Architecture:
  Input → [GIN Conv × L] → Global Pool → MLP Head → Predicted pChEMBL

GIN (Graph Isomorphism Network) is used because it has maximum expressiveness
among message-passing GNNs and captures ring structures well.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool, BatchNorm
    PYG_OK = True
except ImportError:
    PYG_OK = False
    print("⚠  PyTorch Geometric not installed. Using simplified GNN fallback.")

from config import GNN_CONFIG, NUM_ATOM_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# MLP building block
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp(in_dim: int, hidden_dim: int, out_dim: int,
              dropout: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full GNN Model
# ─────────────────────────────────────────────────────────────────────────────

class DrugGNN(nn.Module):
    """
    GIN-based Graph Neural Network for binding affinity regression.

    Inputs:  Data.x, Data.edge_index, Data.batch
    Outputs: scalar pChEMBL prediction per molecule
    """

    def __init__(
        self,
        in_channels    : int   = NUM_ATOM_FEATURES,
        hidden_channels: int   = GNN_CONFIG["hidden_channels"],
        num_layers     : int   = GNN_CONFIG["num_layers"],
        dropout        : float = GNN_CONFIG["dropout"],
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout    = dropout

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # ── GIN layers ───────────────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        for _ in range(num_layers):
            mlp  = build_mlp(hidden_channels, hidden_channels * 2, hidden_channels, dropout)
            if PYG_OK:
                conv = GINConv(mlp, train_eps=True)
            else:
                conv = _FallbackConv(mlp)
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # ── Readout head ─────────────────────────────────────────────────────
        # We concatenate mean + sum pooling (2 × hidden) → dense layers
        readout_dim = hidden_channels * 2
        self.head = nn.Sequential(
            nn.Linear(readout_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),   # scalar output
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection
        x = F.relu(self.input_proj(x))

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout
        if PYG_OK:
            x_mean = global_mean_pool(x, batch)
            x_sum  = global_add_pool(x, batch)
        else:
            x_mean = x_sum = x.mean(dim=0, keepdim=True)

        x = torch.cat([x_mean, x_sum], dim=-1)
        return self.head(x).squeeze(-1)

    def get_embedding(self, data) -> torch.Tensor:
        """
        Return molecule-level embedding (before final head).
        Useful for similarity-based repurposing.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        if PYG_OK:
            x_mean = global_mean_pool(x, batch)
            x_sum  = global_add_pool(x, batch)
        else:
            x_mean = x_sum = x.mean(dim=0, keepdim=True)
        return torch.cat([x_mean, x_sum], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback conv (when PyG not installed)
# ─────────────────────────────────────────────────────────────────────────────

class _FallbackConv(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, x, edge_index):
        # Simple mean aggregation
        src, dst = edge_index
        agg = torch.zeros_like(x)
        count = torch.zeros(x.size(0), 1, device=x.device)
        agg.index_add_(0, dst, x[src])
        count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        return self.mlp(x + agg / count)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GNNTrainer:
    """
    Handles training, validation, and evaluation of DrugGNN.
    """

    def __init__(self, model: DrugGNN, device: str = "cpu"):
        self.model     = model.to(device)
        self.device    = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr           = GNN_CONFIG["learning_rate"],
            weight_decay = 1e-5,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )
        self.criterion = nn.MSELoss()
        self.history   = {"train_loss": [], "val_loss": []}

    def train_epoch(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            loss = self.criterion(pred, batch.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader) -> tuple[float, list, list]:
        """Returns (MSE_loss, predictions, targets)."""
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        preds_all, tgts_all = [], []
        for batch in loader:
            batch  = batch.to(self.device)
            pred   = self.model(batch)
            loss   = self.criterion(pred, batch.y.float())
            total_loss += loss.item()
            n_batches  += 1
            preds_all.extend(pred.cpu().numpy().tolist())
            tgts_all.extend(batch.y.cpu().numpy().tolist())
        return total_loss / max(n_batches, 1), preds_all, tgts_all

    def train(self, train_loader, val_loader,
              epochs: int = GNN_CONFIG["epochs"],
              patience: int = GNN_CONFIG["patience"]) -> dict:
        """Full training loop with early stopping."""
        best_val  = float("inf")
        no_improv = 0
        best_state = None

        print(f"\n{'─'*55}")
        print(f"  Training DrugGNN   |  epochs={epochs}  patience={patience}")
        print(f"{'─'*55}")

        for epoch in range(1, epochs + 1):
            tr_loss        = self.train_epoch(train_loader)
            vl_loss, _, __ = self.evaluate(val_loader)
            self.scheduler.step(vl_loss)
            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}  |  train={tr_loss:.4f}  val={vl_loss:.4f}")

            if vl_loss < best_val - 1e-4:
                best_val  = vl_loss
                no_improv = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                no_improv += 1
                if no_improv >= patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Restore best weights
        if best_state:
            self.model.load_state_dict(best_state)
        print(f"\n  Best val loss: {best_val:.4f}")
        return self.history


# ─────────────────────────────────────────────────────────────────────────────
# Quick model test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = DrugGNN()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DrugGNN | Trainable parameters: {total:,}")
    print(model)
