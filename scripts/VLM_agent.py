import os
import json
import re
import torch
import warnings
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import Counter

# Suppress sklearn 1x1 confusion matrix warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics')

# 🔌 Import your VLM components directly from your existing vlm.py file
from models.vlm import CustomVLM, SimpleTokenizer, generate_caption

# ==========================================
# 1. SEMANTIC NORMALIZATION & METRICS
# ==========================================
def normalize_label(text):
    """Normalizes labels for semantic matching."""
    t = str(text).lower().replace(" ", "").replace("/", "").replace("_", "")
    mapping = {
        'partlycloudy': 'overcast', 
        'residential': 'citystreet', 
        'parkinglot': 'citystreet'
    }
    return mapping.get(t, t)

def calculate_metrics(y_true, y_pred):
    """Calculates accuracy, precision, F1, and Exact/Semantic matches."""
    metrics = {}
    total = len(y_true['weather'])
    if total == 0: return metrics
    
    for category in ['weather', 'timeofday', 'scene']:
        metrics[f'{category}_acc'] = accuracy_score(y_true[category], y_pred[category])
        metrics[f'{category}_prec'] = precision_score(y_true[category], y_pred[category], average='macro', zero_division=0)
        metrics[f'{category}_f1'] = f1_score(y_true[category], y_pred[category], average='macro', zero_division=0)

    exact_matches, semantic_matches = 0, 0
    for i in range(total):
        # Strict Hard Match (Exact String Alignment)
        if (y_true['weather'][i] == y_pred['weather'][i] and
            y_true['timeofday'][i] == y_pred['timeofday'][i] and
            y_true['scene'][i] == y_pred['scene'][i]):
            exact_matches += 1
            
        # Semantic Match (Leniency for related infrastructural/meteorological concepts)
        if (normalize_label(y_pred['weather'][i]) == normalize_label(y_true['weather'][i]) and 
            normalize_label(y_pred['scene'][i]) == normalize_label(y_true['scene'][i]) and 
            y_pred['timeofday'][i] == y_true['timeofday'][i]):
            semantic_matches += 1
            
    metrics['exact_match'] = exact_matches / total
    metrics['semantic_match'] = semantic_matches / total
    return metrics

# ==========================================
# 2. INFERENCE ENGINE
# ==========================================
def extract_label_from_text(text, valid_labels):
    """Robust regex-based parsing to prevent substring overlap errors."""
    text = text.lower()
    for label in valid_labels:
        # Use word boundaries to prevent 'cloudy' triggering inside 'partly cloudy'
        # Handles slashes for 'dawn/dusk' safely
        escaped_label = re.escape(label)
        if re.search(rf'\b{escaped_label}\b', text):
            return label
    return 'undefined'

def run_inference(vlm, tokenizer, transform, DEVICE, img_dir, lbl_dir, map_dict, is_ood=False, max_samples=None):
    """Runs inference and returns true/pred labels and source metadata."""
    y_true = {'weather': [], 'timeofday': [], 'scene': []}
    y_pred = {'weather': [], 'timeofday': [], 'scene': []}
    y_source = []
    y_img = [] 
    
    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if max_samples:
        images = images[:max_samples]
        
    for img_name in tqdm(images, desc=f"Evaluating {'OOD' if is_ood else 'ID'} Data"):
        base_name = os.path.splitext(img_name)[0]
        json_path = os.path.join(lbl_dir, base_name + '.json')
        
        if not os.path.exists(json_path): continue
            
        with open(json_path, 'r') as f:
            attrs = json.load(f).get('attributes', {})
            
        image = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            pred_text = generate_caption(vlm, tokenizer, image_tensor, max_length=18)
            
        y_true['weather'].append(attrs.get('weather', 'undefined'))
        y_true['timeofday'].append(attrs.get('timeofday', 'undefined'))
        y_true['scene'].append(attrs.get('scene', 'undefined'))
        
        y_pred['weather'].append(extract_label_from_text(pred_text, map_dict['weather']))
        y_pred['timeofday'].append(extract_label_from_text(pred_text, map_dict['timeofday']))
        y_pred['scene'].append(extract_label_from_text(pred_text, map_dict['scene']))
        
        source = "_".join(img_name.split('_')[1:-1]).upper() if (is_ood and len(img_name.split('_')) >= 3) else ("UNKNOWN_OOD" if is_ood else "BDD100K_ID")
        y_source.append(source)
        y_img.append(img_name) 
        
    return y_true, y_pred, y_source, y_img

def get_top_errors(true_list, pred_list):
    """Finds top hallucinated pairs, ignoring semantic matches."""
    errors = [f"{t} -> {p}" for t, p in zip(true_list, pred_list) if normalize_label(t) != normalize_label(p)]
    return Counter(errors).most_common(5) # Upped to top 5 for better paper analysis

# ==========================================
# 3. MAIN EVALUATION SCRIPT
# ==========================================
def evaluate_vlm_agent():
    print("🕵️‍♂️ INITIALIZING CONFERENCE-GRADE VLM EVALUATION...")
    
    TRAIN_LBL_DIR = "data/bdd_data/labels/train" 
    WEIGHTS_PATH = "checkpoints/vlm_bdd_checkpoint.pth" 
    ID_TEST_IMG_DIR = "data/bdd_data/images/val" 
    ID_TEST_LBL_DIR = "data/bdd_data/labels/val"
    OOD_TEST_IMG_DIR = "data/ood_data/images"        
    OOD_TEST_LBL_DIR = "data/ood_data/labels"        
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Using device: {DEVICE}")
    
    # 1. Rebuild Vocabulary
    all_captions = []
    for j_file in sorted([f for f in os.listdir(TRAIN_LBL_DIR) if f.endswith('.json')]):
        try:
            with open(os.path.join(TRAIN_LBL_DIR, j_file), 'r') as f:
                attrs = json.load(f).get('attributes', {})
                all_captions.append(f"A driving scene on a {attrs.get('scene', 'unknown')} during {attrs.get('timeofday', 'unknown')} with {attrs.get('weather', 'unknown')} weather")
        except: pass
        
    tokenizer = SimpleTokenizer(all_captions)
    
    # 2. Load Model
    vlm = CustomVLM(vocab_size=tokenizer.vocab_size, llm_embed_dim=256, max_seq_len=128).to(DEVICE)
    vlm.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    vlm.eval()

    MAP = {
        # Sort valid labels by length descending so longer phrases match first (e.g. 'partly cloudy' before 'cloudy')
        'weather': sorted(['clear', 'foggy', 'overcast', 'partly cloudy', 'rainy', 'snowy', 'undefined'], key=len, reverse=True),
        'timeofday': sorted(['dawn/dusk', 'daytime', 'night', 'undefined'], key=len, reverse=True),
        'scene': sorted(['city street', 'gas stations', 'highway', 'parking lot', 'residential', 'tunnel', 'undefined'], key=len, reverse=True)
    }

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. Run Inference
    print("\n🏃‍♂️ PHASE 1: Running IN-DISTRIBUTION (BDD100K) Inference...")
    id_true, id_pred, _, _ = run_inference(vlm, tokenizer, transform, DEVICE, ID_TEST_IMG_DIR, ID_TEST_LBL_DIR, MAP, is_ood=False)
    
    print("\n🌍 PHASE 2: Running OUT-OF-DISTRIBUTION (Curated) Inference...")
    ood_true, ood_pred, ood_source, ood_img = run_inference(vlm, tokenizer, transform, DEVICE, OOD_TEST_IMG_DIR, OOD_TEST_LBL_DIR, MAP, is_ood=True)

    # 4. Calculate Metrics & Export Artifact
    id_metrics = calculate_metrics(id_true, id_pred)
    ood_metrics = calculate_metrics(ood_true, ood_pred)

    evaluation_artifact = {
        "ID_BDD100K": id_metrics,
        "OOD_Curated": ood_metrics,
        "Top_OOD_Hallucinations": {
            "Weather": get_top_errors(ood_true['weather'], ood_pred['weather']),
            "Scene": get_top_errors(ood_true['scene'], ood_pred['scene'])
        }
    }
    with open("evaluation_report.json", "w") as f:
        json.dump(evaluation_artifact, f, indent=4)
    print("\n💾 Saved comprehensive metrics to 'evaluation_report.json'")

    # 5. Terminal Report
    print("\n" + "="*75)
    print(f"{'VLM GENERALIZATION GAP REPORT':^75}")
    print("="*75)
    print(f"{'METRIC':<25} | {'IN-DIST (BDD100K)':<22} | {'OUT-OF-DIST (OOD)':<22}")
    print("-" * 75)
    print(f"{'Weather Accuracy':<25} | {id_metrics['weather_acc']:<22.4f} | {ood_metrics['weather_acc']:<22.4f}")
    print(f"{'Time of Day Acc':<25} | {id_metrics['timeofday_acc']:<22.4f} | {ood_metrics['timeofday_acc']:<22.4f}")
    print(f"{'Scene Accuracy':<25} | {id_metrics['scene_acc']:<22.4f} | {ood_metrics['scene_acc']:<22.4f}")
    print("-" * 75)
    print(f"{'Exact Match (Strict)':<25} | {id_metrics['exact_match']:<22.4f} | {ood_metrics['exact_match']:<22.4f}")
    print(f"{'Exact Match (Semantic)':<25} | {id_metrics['semantic_match']:<22.4f} | {ood_metrics['semantic_match']:<22.4f}")
    print("="*75)

    # 6. Plotting - Academic Bar Chart
    print("\n🎨 Generating Academic-Grade Comparative Bar Chart...")
    sns.set_theme(style="whitegrid", context="paper")
    
    labels = ['Weather', 'Time of Day', 'Scene', 'Strict Match', 'Semantic Match']
    id_scores = [id_metrics['weather_acc'], id_metrics['timeofday_acc'], id_metrics['scene_acc'], id_metrics['exact_match'], id_metrics['semantic_match']]
    ood_scores = [ood_metrics['weather_acc'], ood_metrics['timeofday_acc'], ood_metrics['scene_acc'], ood_metrics['exact_match'], ood_metrics['semantic_match']]

    x = np.arange(len(labels))
    width = 0.35

    fig_bar, ax_bar = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax_bar.bar(x - width/2, id_scores, width, label='In-Distribution (BDD100K)', color='#2c7bb6', edgecolor='black')
    rects2 = ax_bar.bar(x + width/2, ood_scores, width, label='OOD (Curated Dataset)', color='#d7191c', edgecolor='black')

    ax_bar.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax_bar.set_title('Domain Generalization Gap in VLM Environmental Perception', fontsize=14, fontweight='bold', pad=15)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=11)
    ax_bar.legend(loc='upper right', frameon=True, shadow=True)
    ax_bar.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig('phase5_id_vs_ood_comparison.png', dpi=300, bbox_inches='tight')

    # 7. Plotting - OOD Split Confusion Matrices
    print("🎨 Generating Split Confusion Matrices for OOD...")
    unique_sets = sorted(list(set(ood_source)))
    num_sets = len(unique_sets)
    if num_sets == 0: return
    
    fig_cm, axes_cm = plt.subplots(num_sets, 3, figsize=(18, 5 * num_sets), dpi=300)
    fig_cm.suptitle('OOD Domain Generalization by Geographic Region', fontsize=20, fontweight='bold', y=1.02)
    
    if num_sets == 1: axes_cm = np.expand_dims(axes_cm, axis=0)

    categories = ['weather', 'timeofday', 'scene']
    titles = ['Weather Domain', 'Temporal Domain', 'Scene Domain']

    for row_idx, source_set in enumerate(unique_sets):
        indices = [i for i, src in enumerate(ood_source) if src == source_set]
        for col_idx, category in enumerate(categories):
            set_y_true = [ood_true[category][i] for i in indices]
            set_y_pred = [ood_pred[category][i] for i in indices]
            local_labels = sorted(list(set(set_y_true) | set(set_y_pred)))
            if not local_labels: local_labels = ['undefined']
                
            cm = confusion_matrix(set_y_true, set_y_pred, labels=local_labels)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds' if 'OOD' in source_set else 'Blues', 
                        ax=axes_cm[row_idx, col_idx], xticklabels=local_labels, yticklabels=local_labels, 
                        cbar=False, annot_kws={"size": 14, "weight": "bold"}, linewidths=0.5, linecolor='gray')          
            
            axes_cm[row_idx, col_idx].set_title(f"{source_set} | {titles[col_idx]}", fontsize=14, fontweight='bold', pad=10)
            axes_cm[row_idx, col_idx].set_xlabel('Predicted Label', fontsize=12)
            axes_cm[row_idx, col_idx].set_ylabel('Ground Truth', fontsize=12)
            axes_cm[row_idx, col_idx].tick_params(axis='x', rotation=45)
            axes_cm[row_idx, col_idx].tick_params(axis='y', rotation=0)

    plt.tight_layout() 
    plt.savefig('phase5_ood_split_results.png', dpi=300, bbox_inches='tight')
    print("✅ Evaluation complete. Artifacts and plots saved.")

if __name__ == "__main__":
    evaluate_vlm_agent()