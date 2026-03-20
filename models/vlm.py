import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. THE VISION BACKBONE (BDDEyes)
# ==========================================
class BDDEyes(nn.Module):
    """Conference-grade Vision Encoder outputting Spatial Patch Tokens."""
    def __init__(self):
        super().__init__()
        
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()
        self.backbone.avgpool = nn.Identity() 
        
        # ⚠️ RENAMED TO MATCH YOUR CHECKPOINT EXACTLY
        self.shared = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.GELU()
        )

    def forward(self, x):
        features = self.backbone.features(x) 
        B, C, H, W = features.shape
        
        # [Batch, 768, 49] -> [Batch, 49, 768]
        patch_sequence = features.view(B, C, H * W).transpose(1, 2)
        
        # Flatten to 2D so BatchNorm1d doesn't crash: [Batch * 49, 768]
        flat_patches = patch_sequence.reshape(-1, 768)
        
        # Pass through your trained shared layer
        visual_tokens_flat = self.shared(flat_patches)
        
        # Reshape back to 3D sequence: [Batch, 49, 512]
        visual_tokens = visual_tokens_flat.view(B, 49, 512)
        
        return visual_tokens

# ==========================================
# 2. THE LANGUAGE MODEL (MiniLLM)
# ==========================================
class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T).to(x.device)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=4, max_seq_len=128):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

# ==========================================
# 3. THE BRIDGE (CustomVLM)
# ==========================================
class CustomVLM(nn.Module):
    def __init__(self, vocab_size=5000, llm_embed_dim=256, max_seq_len=128):
        super().__init__()
        self.vision = BDDEyes()
        self.llm = MiniLLM(vocab_size, llm_embed_dim, num_heads=8, num_layers=4, max_seq_len=max_seq_len)
        
        # This acts as the final bridge bridging the 512 output of `shared` to the 256 LLM dim
        self.vision_projector = nn.Linear(512, llm_embed_dim)

    def forward(self, images, text_tokens):
        device = images.device
        visual_features = self.vision(images)
        
        # Bridge 512 -> 256
        visual_words = self.vision_projector(visual_features) 
        
        # We don't need unsqueeze(1) anymore because visual_words is already [B, 49, 256]
        text_embeddings = self.llm.token_embedding(text_tokens)
        
        # Concatenate 49 vision tokens + text sequence
        combined_sequence = torch.cat([visual_words, text_embeddings], dim=1)
        
        seq_length = combined_sequence.size(1)
        positions = torch.arange(0, seq_length, dtype=torch.long, device=device)
        pos_emb = self.llm.position_embedding(positions)
        
        x = combined_sequence + pos_emb
        x = self.llm.blocks(x)
        x = self.llm.ln_f(x)
        
        logits = self.llm.lm_head(x) 
        return logits

# ==========================================
# 4. THE TOKENIZER 
# ==========================================
class SimpleTokenizer:
    def __init__(self, text_dataset):
        print("🔤 Building vocabulary...")
        all_words = " ".join(text_dataset).lower().split()
        unique_words = sorted(list(set(all_words)))
        
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        
        self.word2int = {w: i+3 for i, w in enumerate(unique_words)}
        self.int2word = {i+3: w for i, w in enumerate(unique_words)}
        
        self.vocab_size = len(self.word2int) + 3 

    def encode(self, text, max_len=15):
        words = text.lower().split()
        tokens = [self.bos_token] + [self.word2int.get(w, self.pad_token) for w in words] + [self.eos_token]
        
        if len(tokens) < max_len:
            tokens = tokens + [self.pad_token] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len-1] + [self.eos_token]
            
        return torch.tensor(tokens, dtype=torch.long)

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            
        words = []
        for t in tokens:
            if t == self.eos_token:
                break
            if t not in [self.pad_token, self.bos_token]:
                words.append(self.int2word.get(t, "<UNK>"))
        return " ".join(words)

# ==========================================
# 5. THE TRAINING LOOP
# ==========================================
def train_vlm(vlm, images, target_tokens, epochs=30):
    for param in vlm.vision.parameters():
        param.requires_grad = False
        
    trainable_params = [p for p in vlm.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    
    vlm.train()
    print("\n🔥 Starting Training Loop...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = vlm(images, target_tokens)
        
        predictions = logits[:, :-1, :].contiguous() 
        predictions = predictions.view(-1, vlm.llm.lm_head.out_features)
        targets = target_tokens.view(-1)
        
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {loss.item():.4f}")

    return vlm

# ==========================================
# 6. INFERENCE 
# ==========================================
def generate_caption(vlm, tokenizer, image, max_length=15):
    vlm.eval()
    device = image.device
    
    generated_tokens = []
    current_sequence = torch.tensor([[tokenizer.bos_token]], dtype=torch.long, device=device)
    
    print("\n🤖 VLM is thinking...")
    with torch.no_grad():
        for _ in range(max_length):
            logits = vlm(image, current_sequence)
            next_word_logits = logits[0, -1, :] 
            next_word_id = torch.argmax(next_word_logits).item()
            
            if next_word_id == tokenizer.eos_token:
                break
                
            generated_tokens.append(next_word_id)
            new_token_tensor = torch.tensor([[next_word_id]], dtype=torch.long, device=device)
            current_sequence = torch.cat([current_sequence, new_token_tensor], dim=1)
            
    return tokenizer.decode(generated_tokens)

# ==========================================
# 7. THE DATALOADER (1 IMAGE = 1 JSON)
# ==========================================
class BDDTextDataset(Dataset):
    def __init__(self, labels_dir, image_dir, tokenizer, max_seq_len=18):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Grab all the image filenames in the directory
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📂 Found {len(self.image_files)} images in {image_dir}")
            
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0] # e.g., 'fdb23892-929ab524'
        
        # Construct exact paths
        img_path = os.path.join(self.image_dir, img_name)
        json_path = os.path.join(self.labels_dir, base_name + '.json')
        
        # 1. Load Image
        try:
            image = Image.open(img_path).convert("RGB")
            image_tensor = self.transform(image)
        except Exception:
            # If image is broken, return a blank tensor so training doesn't crash
            image_tensor = torch.zeros((3, 224, 224))
            
        # 2. Load matching JSON
        try:
            with open(json_path, 'r') as f:
                item = json.load(f)
            attrs = item.get('attributes', {})
        except Exception:
            # Fallback if a JSON is missing
            attrs = {}

        weather = attrs.get('weather', 'unknown')
        scene = attrs.get('scene', 'unknown')
        timeofday = attrs.get('timeofday', 'unknown')
        
        # Build the sentence
        caption = f"A driving scene on a {scene} during {timeofday} with {weather} weather"
        token_tensor = self.tokenizer.encode(caption, max_len=self.max_seq_len)
        
        return image_tensor, token_tensor

# ==========================================
# 8. THE MAIN VLM TRAINING LOOP
# ==========================================
def main_train_loop():
    print("🚀 INITIALIZING VLM TRAINING...")
    
    # --- YOUR TRAINING PATHS ---
    # Make sure these point to your training set, not the test set!
    TRAIN_IMG_DIR = "../data/bdd_data/images/train"
    TRAIN_LBL_DIR = "../data/bdd_data/labels/train"
    
    # Auto-detect Apple Silicon (MPS), CUDA, or CPU
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {DEVICE}")

    # 1. Build Vocabulary from Training Data
    print("\n🔍 Scanning training JSONs to build vocabulary...")
    all_captions = []
    json_files = sorted([f for f in os.listdir(TRAIN_LBL_DIR) if f.endswith('.json')])
    for j_file in json_files:
        try:
            with open(os.path.join(TRAIN_LBL_DIR, j_file), 'r') as f:
                attrs = json.load(f).get('attributes', {})
                caption = f"A driving scene on a {attrs.get('scene', 'unknown')} during {attrs.get('timeofday', 'unknown')} with {attrs.get('weather', 'unknown')} weather"
                all_captions.append(caption)
        except Exception:
            continue
            
    tokenizer = SimpleTokenizer(all_captions)
    print(f"✅ Vocabulary Size: {tokenizer.vocab_size} words")

    # 2. Setup Dataloader
    train_dataset = BDDTextDataset(TRAIN_LBL_DIR, TRAIN_IMG_DIR, tokenizer, max_seq_len=18)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # 3. Initialize the VLM
    vlm = CustomVLM(vocab_size=tokenizer.vocab_size, llm_embed_dim=256, max_seq_len=128).to(DEVICE)
    
    # Plug in your trained "Eyes" from your old MTL script
    weights_path = "../checkpoints/best_convnext_mtl.pth" 
    if os.path.exists(weights_path):
        try:
            saved_state_dict = torch.load(weights_path, map_location=DEVICE)
            # We use strict=False because the saved weights include the old classification heads 
            # which we deleted. PyTorch will just load the backbone!
            vlm.vision.load_state_dict(saved_state_dict, strict=False)
            print(f"🧠 Successfully loaded trained vision weights from {weights_path}!")
        except Exception as e:
            print(f"⚠️ Could not load eyes: {e}")
    else:
        print("⚠️ No pre-trained vision weights found. Starting fresh!")

    # 🔥 OPTION B: PARTIAL UNFREEZING!
    # 1. Freeze the core ConvNeXt backbone (keeps basic edge/shape detection)
    for param in vlm.vision.backbone.parameters():
        param.requires_grad = False
        
    # 2. UNFREEZE the custom `shared` projection head so it can learn language!
    for param in vlm.vision.shared.parameters():
        param.requires_grad = True

    # 4. Optimizer and Loss (Differential Learning Rates)
    # We teach the LLM at normal speed, but tweak the vision head very slowly
    optimizer = optim.AdamW([
        {'params': vlm.vision.shared.parameters(), 'lr': 1e-5},    # 🐢 Slow tweaks to vision
        {'params': vlm.vision_projector.parameters(), 'lr': 1e-4}, # 🐇 Normal speed
        {'params': vlm.llm.parameters(), 'lr': 1e-4}               # 🐇 Normal speed
    ], weight_decay=1e-4)
    
    # CRITICAL: We tell the loss function to ignore the PAD token (0). 
    # Otherwise, it gets a "low loss" just by guessing empty space over and over.
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token) 

    # 5. The Training Loop
    EPOCHS = 30
    print(f"\n🔥 Starting Epochs...")
    
    for epoch in range(EPOCHS):
        vlm.train()
        total_loss = 0
        
        for batch_idx, (images, tokens) in enumerate(train_loader):
            images, tokens = images.to(DEVICE), tokens.to(DEVICE)
            
            optimizer.zero_grad()
            logits = vlm(images, tokens)
            
            # --- 🛠️ THE CRITICAL FIX: Slicing the correct logits ---
            # logits is [Batch, 67, VocabSize]
            # We slice from index 48 to 65 to grab exactly the 18 predictions for the text
            predictions = logits[:, 48:-1, :].contiguous().view(-1, tokenizer.vocab_size)
            targets = tokens.contiguous().view(-1)
            
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Print an update every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"⭐ Epoch {epoch+1} Complete | Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(vlm.state_dict(), "vlm_bdd_checkpoint.pth")
        
        # --- FUN PART: WATCH IT LEARN ---
        # At the end of every epoch, let's ask it to caption the very last image it saw!
        vlm.eval()
        with torch.no_grad():
            sample_image = images[0].unsqueeze(0) # Grab the first image from the last batch
            print("\n👀 Testing generative output on a sample image...")
            pred_text = generate_caption(vlm, tokenizer, sample_image, max_length=15)
            target_text = tokenizer.decode(tokens[0])
            
            print(f"🎯 Target string: {target_text}")
            print(f"🤖 VLM generated: '{pred_text}'\n")

if __name__ == "__main__":
    main_train_loop()