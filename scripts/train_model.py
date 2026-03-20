import os
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION (THE FIREWALL)
# ==========================================
class Config:
    # Strict Paths for Training and Validation (NO TEST DATA HERE)
    TRAIN_IMG_DIR = "bdd_data/images/train"
    TRAIN_LBL_DIR = "bdd_data/labels/train"
    VAL_IMG_DIR = "bdd_data/images/val"
    VAL_LBL_DIR = "bdd_data/labels/val"
    
    # PKL Cache Paths
    TRAIN_PKL = "bdd_train_cache.pkl"
    VAL_PKL = "bdd_val_cache.pkl"
    
    # Model Output
    BEST_MODEL_OUT = "best_convnext_mtl.pth"
    
    # Hyperparameters
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = 224
    BATCH_SIZE = 64
    EPOCHS = 10
    BASE_LR = 1e-4

class Color:
    GREEN, CYAN, BOLD, YELLOW, END = '\033[92m', '\033[96m', '\033[1m', '\033[93m', '\033[0m'

# Canonical BDD100K Mappings
MAP = {
    'weather': ['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined'],
    'timeofday': ['dawn/dusk', 'daytime', 'night', 'undefined'],
    'scene': ['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined']
}

# ==========================================
# 2. PHASE 1: PKL COMPILER
# ==========================================
def build_pkl_cache(img_dir, lbl_dir, out_pkl_path):
    """Scans directories, parses JSONs, and builds a fast-loading .pkl index."""
    if os.path.exists(out_pkl_path):
        print(f"{Color.CYAN}📦 Found existing cache: {out_pkl_path}{Color.END}")
        return

    print(f"{Color.YELLOW}⚙️ Building PKL cache for {img_dir}...{Color.END}")
    cache = []
    images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    
    for img_name in tqdm(images, desc="Compiling JSONs"):
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.json'))
        
        if not os.path.exists(lbl_path):
            continue 
            
        with open(lbl_path, 'r') as f:
            attr = json.load(f)['attributes']
            
        cache.append({
            'img_path': img_path,
            'w': MAP['weather'].index(attr['weather']),
            't': MAP['timeofday'].index(attr['timeofday']),
            's': MAP['scene'].index(attr['scene'])
        })
        
    with open(out_pkl_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f"{Color.GREEN}✅ Saved {len(cache)} records to {out_pkl_path}{Color.END}\n")

# ==========================================
# 3. FAST DATASET & MODEL
# ==========================================
class BDDPklDataset(Dataset):
    def __init__(self, pkl_path, transform=None):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['img_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, item['w'], item['t'], item['s']

class BDDEyes(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        
        self.shared = nn.Sequential(nn.Linear(768, 512), nn.BatchNorm1d(512), nn.GELU())
        self.w_head = nn.Linear(512, len(MAP['weather']))
        self.t_head = nn.Linear(512, len(MAP['timeofday']))
        self.s_head = nn.Linear(512, len(MAP['scene']))

    def forward(self, x):
        feat = self.shared(torch.flatten(self.pooling(self.backbone(x)), 1))
        return self.w_head(feat), self.t_head(feat), self.s_head(feat)

# --- NEW: Academic Multi-Task Loss Weighting (Kendall et al. CVPR 2018) ---
class HomoscedasticUncertaintyLoss(nn.Module):
    """Automatically scales multi-task losses based on task uncertainty."""
    def __init__(self, num_tasks=3):
        super().__init__()
        # Learnable log-variances for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            # precision = exp(-log_var)
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss

# ==========================================
# 4. PHASE 2: RIGOROUS TRAINING LOOP
# ==========================================
def train_model():
    print(f"{Color.BOLD}🚀 INITIALIZING RIGOROUS MULTI-TASK TRAINING...{Color.END}")
    
    build_pkl_cache(Config.TRAIN_IMG_DIR, Config.TRAIN_LBL_DIR, Config.TRAIN_PKL)
    build_pkl_cache(Config.VAL_IMG_DIR, Config.VAL_LBL_DIR, Config.VAL_PKL)
    
    train_tf = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Added slight jitter for robustness
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_tf = transforms.Compose([
        transforms.Resize((Config.INPUT_SIZE, Config.INPUT_SIZE)),
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_loader = DataLoader(BDDPklDataset(Config.TRAIN_PKL, train_tf), batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(BDDPklDataset(Config.VAL_PKL, val_tf), batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    model = BDDEyes().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    mtl_loss_mixer = HomoscedasticUncertaintyLoss(num_tasks=3).to(Config.DEVICE)
    
    
    
    # --- NEW: Differential Learning Rates ---
    # The pre-trained backbone learns 10x slower than the fresh heads to prevent catastrophic forgetting
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': Config.BASE_LR * 0.1},
        {'params': model.shared.parameters(), 'lr': Config.BASE_LR},
        {'params': model.w_head.parameters(), 'lr': Config.BASE_LR},
        {'params': model.t_head.parameters(), 'lr': Config.BASE_LR},
        {'params': model.s_head.parameters(), 'lr': Config.BASE_LR},
        {'params': mtl_loss_mixer.parameters(), 'lr': Config.BASE_LR * 10} # Let uncertainty parameters adapt quickly
    ], weight_decay=1e-4)
    
    # --- NEW: Cosine Annealing Scheduler ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    best_val_score = 0.0
    
    # 5. The Epoch Loop
    for epoch in range(Config.EPOCHS):
        print(f"\n{Color.BOLD}--- Epoch {epoch+1}/{Config.EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e} ---{Color.END}")
        
        # --- TRAINING ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        
        for imgs, w, t, s in pbar:
            imgs = imgs.to(Config.DEVICE, non_blocking=True)
            w, t, s = w.to(Config.DEVICE), t.to(Config.DEVICE), s.to(Config.DEVICE)
            
            optimizer.zero_grad()
            out_w, out_t, out_s = model(imgs)
            
            # Calculate individual losses
            l_w = criterion(out_w, w)
            l_t = criterion(out_t, t)
            l_s = criterion(out_s, s)
            
            # Dynamically weight them based on learned uncertainty
            loss = mtl_loss_mixer([l_w, l_t, l_s])
            
            loss.backward()
            
            # --- NEW: Gradient Clipping for stability ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            
        scheduler.step() # Step the learning rate down the cosine curve
            
        # --- VALIDATION (THE FIREWALL) ---
        model.eval()
        val_perfect, total_imgs = 0, 0
        
        with torch.no_grad():
            for imgs, w, t, s in tqdm(val_loader, desc="Validating"):
                imgs = imgs.to(Config.DEVICE, non_blocking=True)
                w, t, s = w.to(Config.DEVICE), t.to(Config.DEVICE), s.to(Config.DEVICE)
                
                out_w, out_t, out_s = model(imgs)
                pred_w, pred_t, pred_s = out_w.argmax(1), out_t.argmax(1), out_s.argmax(1)
                
                matches = (pred_w == w).int() + (pred_t == t).int() + (pred_s == s).int()
                val_perfect += (matches == 3).sum().item()
                total_imgs += imgs.size(0)
                
        val_score = val_perfect / total_imgs
        print(f"Validation Perfect Match (3/3): {Color.CYAN}{val_score:.2%}{Color.END}")
        
        # Print learned task weights just for interesting logging
        with torch.no_grad():
            weights = torch.exp(-mtl_loss_mixer.log_vars).cpu().numpy()
            print(f"Learned Task Weights -> Weather: {weights[0]:.2f}, Time: {weights[1]:.2f}, Scene: {weights[2]:.2f}")
        
        if val_score > best_val_score:
            best_val_score = val_score
            # We save the model weights to be consumed by the VLM later
            torch.save(model.state_dict(), Config.BEST_MODEL_OUT)
            print(f"{Color.GREEN}⭐ New Best Vision Backbone Saved to {Config.BEST_MODEL_OUT}{Color.END}")

if __name__ == "__main__":
    train_model()