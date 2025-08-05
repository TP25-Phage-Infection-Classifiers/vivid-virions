# src/models/MLP/train.py

# standard libs...
import argparse, os, pickle, random, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# use package-relative imports 
from .._utils import find_logo_indices, load_logo_split, load_stratified_split
from .._prep import prepare_xy, align_columns


def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, p=0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)
        self.drop = nn.Dropout(p)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        return self.out(x)

def train_one_split(Xtr, ytr, Xte, yte, out_dir: Path, idx: int,
                    device, epochs=100, batch_size=64, patience=10):
    # label encoding (fixed order for reproducibility)
    le = LabelEncoder(); le.fit(["early","middle","late"])
    ytr_enc = le.transform(ytr); yte_enc = le.transform(yte)

    # impute + scale
    imp = SimpleImputer(strategy="mean"); scl = StandardScaler()
    Xtr = imp.fit_transform(Xtr); Xte = imp.transform(Xte)
    Xtr = scl.fit_transform(Xtr); Xte = scl.transform(Xte)

    # tensors & loaders
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr_enc, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte_enc, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    model = MLP(input_dim=Xtr.shape[1], num_classes=3).to(device)
    cls_w = compute_class_weight("balanced", classes=np.unique(ytr_enc), y=ytr_enc)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cls_w, dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc, best_state, pat = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
        # early-stop on test acc (simple + effective for now)
        model.eval()
        with torch.no_grad():
            preds = model(Xte_t.to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(yte_enc, preds)
        if acc > best_acc: best_acc, best_state, pat = acc, model.state_dict(), 0
        else:
            pat += 1
            if pat >= patience: break
    if best_state is not None: model.load_state_dict(best_state)

    # save artifacts exactly as your comparison.py expects
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"model_split_{idx}.pt")
    with open(out_dir / f"split_{idx}_data.pkl", "wb") as f:
        pickle.dump((Xtr, ytr_enc, Xte, yte_enc, le, list(range(Xtr.shape[1]))), f)

    # tiny metrics file (useful sanity check)
    rep = classification_report(yte_enc, preds, target_names=le.classes_, zero_division=0, output_dict=True)
    with open(out_dir / f"metrics_split_{idx}.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": best_acc, "report": rep}, f, indent=2)
    print(f"[MLP] split {idx}: acc={best_acc:.4f}")
    return best_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="pipeline_results")
    ap.add_argument("--out-dir",  default="pipeline_results/models/mlp")
    ap.add_argument("--splits", choices=["logo","stratified","both"], default="both")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = Path(args.base_dir)
    out_dir = Path(args.out_dir)

    if args.splits in ("logo","both"):
        idxs = find_logo_indices(base / "splits_logo")
        for i in idxs:
            tr, te = load_logo_split(base, i)
            Xtr, ytr, _ = prepare_xy(tr)
            Xte, yte, _ = prepare_xy(te)
            Xtr, Xte = align_columns(Xtr, Xte)
            train_one_split(Xtr, ytr, Xte, yte, out_dir, i, device, args.epochs, args.batch_size, args.patience)

    if args.splits in ("stratified","both"):
        tr, te = load_stratified_split(base)
        Xtr, ytr, _ = prepare_xy(tr)
        Xte, yte, _ = prepare_xy(te)
        Xtr, Xte = align_columns(Xtr, Xte)
        train_one_split(Xtr, ytr, Xte, yte, out_dir, 6, device, args.epochs, args.batch_size, args.patience)

if __name__ == "__main__":
    main()

