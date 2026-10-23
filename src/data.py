"""
Data Pipeline for Indian Medicinal Plant Leaf Identification (PS-10)
Handles data loading, perceptual hash duplicate detection, leak-free stratified splits,
Albumentations/Torchvision augmentations, and PyTorch Datasets.
"""

import os
import glob
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import imagehash
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Botanical species metadata for the 30 classes in Mendeley Medicinal Leaf Dataset
SPECIES_METADATA = {
    "Alpinia Galanga (Rasna)": {
        "botanical_name": "Alpinia galanga",
        "common_name": "Rasna / Greater Galangal",
        "family": "Zingiberaceae",
        "medicinal_use": "Anti-inflammatory, cures rheumatism, bronchodilator, carminative."
    },
    "Amaranthus Viridis (Arive-Dantu)": {
        "botanical_name": "Amaranthus viridis",
        "common_name": "Arive-Dantu / Slender Amaranth",
        "family": "Amaranthaceae",
        "medicinal_use": "Diuretic, laxative, treats respiratory issues, rich in iron/antioxidants."
    },
    "Artocarpus Heterophyllus (Jackfruit)": {
        "botanical_name": "Artocarpus heterophyllus",
        "common_name": "Halasu / Jackfruit",
        "family": "Moraceae",
        "medicinal_use": "Hypoglycemic (diabetes management), wound healing, antimicrobial."
    },
    "Azadirachta Indica (Neem)": {
        "botanical_name": "Azadirachta indica",
        "common_name": "Neem / Bevu",
        "family": "Meliaceae",
        "medicinal_use": "Powerful antifungal, antibacterial, blood purifier, dental health."
    },
    "Basella Alba (Basale)": {
        "botanical_name": "Basella alba",
        "common_name": "Basale / Malabar Spinach",
        "family": "Basellaceae",
        "medicinal_use": "Treats mouth ulcers, digestive cooling, rich in vitamins A, C and iron."
    },
    "Brassica Juncea (Indian Mustard)": {
        "botanical_name": "Brassica juncea",
        "common_name": "Sasive / Indian Mustard",
        "family": "Brassicaceae",
        "medicinal_use": "Relieves muscle spasms, antimicrobial, stimulates appetite and digestion."
    },
    "Carissa Carandas (Karanda)": {
        "botanical_name": "Carissa carandas",
        "common_name": "Karanda / Kavali",
        "family": "Apocynaceae",
        "medicinal_use": "Antiscorbutic, cardiotonic, treats acidity, indigestion, fever."
    },
    "Citrus Limon (Lemon)": {
        "botanical_name": "Citrus limon",
        "common_name": "Nimbe / Lemon",
        "family": "Rutaceae",
        "medicinal_use": "High Vitamin C, treats scurvy, antimicrobial, antioxidant, supports digestion."
    },
    "Ficus Auriculata (Roxburgh fig)": {
        "botanical_name": "Ficus auriculata",
        "common_name": "Roxburgh Fig / Elephant Ear Fig",
        "family": "Moraceae",
        "medicinal_use": "Treats diarrhoea, dysentery, cuts, wounds, and stomach ailments."
    },
    "Ficus Religiosa (Peepal Tree)": {
        "botanical_name": "Ficus religiosa",
        "common_name": "Peepal / Arali",
        "family": "Moraceae",
        "medicinal_use": "Cardiovascular tonic, cures asthma, anti-diabetic, relieves skin diseases."
    },
    "Hibiscus Rosa-sinensis": {
        "botanical_name": "Hibiscus rosa-sinensis",
        "common_name": "Dasavala / China Rose",
        "family": "Malvaceae",
        "medicinal_use": "Hair growth stimulant, anti-hypertensive, regulates menstrual cycle."
    },
    "Jasminum (Jasmine)": {
        "botanical_name": "Jasminum officinale / sambac",
        "common_name": "Mallige / Jasmine",
        "family": "Oleaceae",
        "medicinal_use": "Calming sedative, antiseptic, relieves headaches, treats dermatosis."
    },
    "Mangifera Indica (Mango)": {
        "botanical_name": "Mangifera indica",
        "common_name": "Mavu / Mango",
        "family": "Anacardiaceae",
        "medicinal_use": "Tender leaves treat diabetes, anti-diarrheal, reduces blood pressure."
    },
    "Mentha (Mint)": {
        "botanical_name": "Mentha spicata / piperita",
        "common_name": "Pudina / Mint",
        "family": "Lamiaceae",
        "medicinal_use": "Relieves indigestion, IBS, respiratory decongestant, cooling antiseptic."
    },
    "Moringa Oleifera (Drumstick)": {
        "botanical_name": "Moringa oleifera",
        "common_name": "Nugge / Drumstick",
        "family": "Moringaceae",
        "medicinal_use": "Nutritional powerhouse, anti-inflammatory, regulates glucose and lipids."
    },
    "Muntingia Calabura (Jamaica Cherry-Gasagase)": {
        "botanical_name": "Muntingia calabura",
        "common_name": "Gasagase Hannu / Jamaica Cherry",
        "family": "Muntingiaceae",
        "medicinal_use": "Antipruritic, relieves headaches, anti-inflammatory, cardioprotective."
    },
    "Murraya Koenigii (Curry)": {
        "botanical_name": "Murraya koenigii",
        "common_name": "Karibevu / Curry Leaf",
        "family": "Rutaceae",
        "medicinal_use": "Supports hair health, antidiabetic, hepatoprotective, lowers cholesterol."
    },
    "Nerium Oleander (Oleander)": {
        "botanical_name": "Nerium oleander",
        "common_name": "Kanagile / Oleander (Caution: Toxic look-alike!)",
        "family": "Apocynaceae",
        "medicinal_use": "Cardiotonic in extreme micro-doses; HIGHLY TOXIC if misused. Vital safety check!"
    },
    "Nyctanthes Arbor-tristis (Parijata)": {
        "botanical_name": "Nyctanthes arbor-tristis",
        "common_name": "Parijata / Night Jasmine",
        "family": "Oleaceae",
        "medicinal_use": "Potent anti-arthritic, cures sciatica, treats chronic fevers, anti-malarial."
    },
    "Ocimum Tenuiflorum (Tulsi)": {
        "botanical_name": "Ocimum tenuiflorum / sanctum",
        "common_name": "Tulsi / Holy Basil",
        "family": "Lamiaceae",
        "medicinal_use": "Adaptogen, boosts immunity, cures cough/colds, antistress, antimicrobial."
    },
    "Piper Betle (Betel)": {
        "botanical_name": "Piper betle",
        "common_name": "Vilyadhele / Betel Leaf",
        "family": "Piperaceae",
        "medicinal_use": "Carminative, antimicrobial breath purifier, digestive stimulant, wound healing."
    },
    "Plectranthus Amboinicus (Mexican Mint)": {
        "botanical_name": "Plectranthus amboinicus",
        "common_name": "Dodda Pathre / Indian/Mexican Mint",
        "family": "Lamiaceae",
        "medicinal_use": "Cures pediatric cold/cough, asthma relief, bronchodilator, digestive carminative."
    },
    "Pongamia Pinnata (Indian Beech)": {
        "botanical_name": "Pongamia pinnata",
        "common_name": "Honge / Indian Beech",
        "family": "Fabaceae",
        "medicinal_use": "Treats eczema, leprosy, wounds, rheumatologic pain relief."
    },
    "Psidium Guajava (Guava)": {
        "botanical_name": "Psidium guajava",
        "common_name": "Sebe / Guava",
        "family": "Myrtaceae",
        "medicinal_use": "Astringent, treats dysentery, toothache relief, supports glycemic control."
    },
    "Punica Granatum (Pomegranate)": {
        "botanical_name": "Punica granatum",
        "common_name": "Dalimbe / Pomegranate",
        "family": "Lythraceae",
        "medicinal_use": "Astringent, cures stomatitis, antioxidant, treats enteric infections."
    },
    "Santalum Album (Sandalwood)": {
        "botanical_name": "Santalum album",
        "common_name": "Gandha / Sandalwood",
        "family": "Santalaceae",
        "medicinal_use": "Skin cooling antiseptic, relieves burning sensations, sedative, aromatic."
    },
    "Syzygium Cumini (Jamun)": {
        "botanical_name": "Syzygium cumini",
        "common_name": "Nerale / Jamun / Black Plum",
        "family": "Myrtaceae",
        "medicinal_use": "Anti-hyperglycemic, treats chronic diarrhea, oral astringent."
    },
    "Syzygium Jambos (Rose Apple)": {
        "botanical_name": "Syzygium jambos",
        "common_name": "Pannerale / Rose Apple",
        "family": "Myrtaceae",
        "medicinal_use": "Antimicrobial, febrifuge, treats digestive tract inflammation."
    },
    "Tabernaemontana Divaricata (Crape Jasmine)": {
        "botanical_name": "Tabernaemontana divaricata",
        "common_name": "Nandi Battalu / Crape Jasmine",
        "family": "Apocynaceae",
        "medicinal_use": "Ophthalmic medicine for eye irritation, pain relief, antimicrobial."
    },
    "Trigonella Foenum-graecum (Fenugreek)": {
        "botanical_name": "Trigonella foenum-graecum",
        "common_name": "Menthya / Fenugreek",
        "family": "Fabaceae",
        "medicinal_use": "Lowers blood glucose, stimulates lactation, digestive soothe."
    }
}

CLASS_NAMES = sorted(list(SPECIES_METADATA.keys()))
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)}


def scan_dataset(data_dir: str) -> pd.DataFrame:
    """Scans dataset directory, validates image formats and removes corrupt files."""
    records = []
    for cls_name in sorted(os.listdir(data_dir)):
        cls_folder = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_folder):
            continue
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for fname in os.listdir(cls_folder):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_exts:
                continue
            fpath = os.path.join(cls_folder, fname)
            try:
                with Image.open(fpath) as img:
                    img.verify()
                # Check opening properly
                with Image.open(fpath) as img:
                    w, h = img.size
                    mode = img.mode
                records.append({
                    "filepath": fpath,
                    "filename": fname,
                    "class_name": cls_name,
                    "label": CLASS_TO_IDX[cls_name],
                    "width": w,
                    "height": h,
                    "channels": 3 if mode == "RGB" else (1 if mode == "L" else len(mode)),
                    "format": ext.replace(".", "").upper()
                })
            except Exception as e:
                print(f"Skipping corrupt image: {fpath} ({e})")
    df = pd.DataFrame(records)
    return df


def compute_perceptual_hashes(df: pd.DataFrame, hash_size: int = 8) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Computes perceptual hashes (pHash) and finds duplicate/near-duplicate image clusters."""
    hashes = []
    for fpath in df["filepath"]:
        with Image.open(fpath) as img:
            phash_val = str(imagehash.phash(img.convert("RGB"), hash_size=hash_size))
            hashes.append(phash_val)
    df["phash"] = hashes
    
    # Group near duplicates using Hamming distance <= 4
    n = len(df)
    phash_objs = [imagehash.hex_to_hash(h) for h in df["phash"]]
    clusters = {}
    visited = set()
    
    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j not in visited and df.iloc[i]["class_name"] == df.iloc[j]["class_name"]:
                if phash_objs[i] - phash_objs[j] <= 4:
                    group.append(j)
                    visited.add(j)
        if len(group) > 1:
            leader = df.iloc[i]["phash"]
            clusters[leader] = [df.iloc[idx]["filepath"] for idx in group]

    return df, clusters


def create_leak_free_splits(
    df: pd.DataFrame,
    clusters: Dict[str, List[str]],
    splits_dir: str = "splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates stratified Train/Val/Test splits with 0% data leakage.
    Images within the same perceptual hash duplicate cluster are grouped strictly
    into the same split partition.
    """
    os.makedirs(splits_dir, exist_ok=True)
    
    # Map each file to a cluster group ID
    filepath_to_group = {}
    group_counter = 0
    for leader, filepaths in clusters.items():
        for fp in filepaths:
            filepath_to_group[fp] = f"grp_{group_counter}"
        group_counter += 1
    
    # Assign unique group IDs to unclustered files
    group_ids = []
    for fp in df["filepath"]:
        if fp in filepath_to_group:
            group_ids.append(filepath_to_group[fp])
        else:
            group_ids.append(f"grp_{group_counter}")
            group_counter += 1
    df["group_id"] = group_ids

    # Group dataframe by unique cluster items
    group_df = df.groupby("group_id").agg({
        "class_name": "first",
        "label": "first",
        "filepath": list
    }).reset_index()

    # Stratified split on groups
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_grp_idx, temp_grp_idx = next(sss1.split(group_df, group_df["label"]))

    train_groups = group_df.iloc[train_grp_idx]
    temp_groups = group_df.iloc[temp_grp_idx].reset_index(drop=True)

    val_prop = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_prop), random_state=seed)
    val_grp_idx, test_grp_idx = next(sss2.split(temp_groups, temp_groups["label"]))

    val_groups = temp_groups.iloc[val_grp_idx]
    test_groups = temp_groups.iloc[test_grp_idx]

    train_files = set([fp for sublist in train_groups["filepath"] for fp in sublist])
    val_files = set([fp for sublist in val_groups["filepath"] for fp in sublist])
    test_files = set([fp for sublist in test_groups["filepath"] for fp in sublist])

    train_df = df[df["filepath"].isin(train_files)].copy().reset_index(drop=True)
    val_df = df[df["filepath"].isin(val_files)].copy().reset_index(drop=True)
    test_df = df[df["filepath"].isin(test_files)].copy().reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Save to CSV for persistent reproducibility
    train_df.to_csv(os.path.join(splits_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test_split.csv"), index=False)

    # 5-fold CV on train + val pool
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_df["cv_fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train_val_df, train_val_df["label"])):
        train_val_df.loc[val_idx, "cv_fold"] = fold
    train_val_df.to_csv(os.path.join(splits_dir, "cv_5folds.csv"), index=False)

    return train_df, val_df, test_df


def verify_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, bool]:
    """Verifies that filename overlap and perceptual hash overlap across splits are strictly zero."""
    train_fps = set(train_df["filepath"])
    val_fps = set(val_df["filepath"])
    test_fps = set(test_df["filepath"])

    file_overlap_tv = len(train_fps.intersection(val_fps))
    file_overlap_tt = len(train_fps.intersection(test_fps))
    file_overlap_vt = len(val_fps.intersection(test_fps))

    train_hashes = set(train_df["phash"])
    val_hashes = set(val_df["phash"])
    test_hashes = set(test_df["phash"])

    hash_overlap_tv = len(train_hashes.intersection(val_hashes))
    hash_overlap_tt = len(train_hashes.intersection(test_hashes))
    hash_overlap_vt = len(val_hashes.intersection(test_hashes))

    leakage_report = {
        "zero_file_leakage": (file_overlap_tv == 0 and file_overlap_tt == 0 and file_overlap_vt == 0),
        "zero_hash_leakage": (hash_overlap_tv == 0 and hash_overlap_tt == 0 and hash_overlap_vt == 0),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "file_overlap_train_val": file_overlap_tv,
        "file_overlap_train_test": file_overlap_tt,
        "file_overlap_val_test": file_overlap_vt,
        "hash_overlap_train_val": hash_overlap_tv,
        "hash_overlap_train_test": hash_overlap_tt,
        "hash_overlap_val_test": hash_overlap_vt
    }
    return leakage_report


def compute_train_normalization(train_df: pd.DataFrame, img_size: int = 224, sample_limit: int = 500) -> Tuple[List[float], List[float]]:
    """Calculates dataset RGB mean and standard deviation strictly on the training partition."""
    sample_fps = train_df["filepath"].sample(n=min(sample_limit, len(train_df)), random_state=42).tolist()
    mean = np.zeros(3)
    std = np.zeros(3)
    total_pixels = 0

    for fp in sample_fps:
        with Image.open(fp) as img:
            arr = np.array(img.convert("RGB").resize((img_size, img_size)), dtype=np.float32) / 255.0
            mean += arr.mean(axis=(0, 1))
            std += arr.std(axis=(0, 1))
            total_pixels += 1

    mean = (mean / total_pixels).tolist()
    std = (std / total_pixels).tolist()
    return mean, std


def get_transforms(mode: str = "strong", img_size: int = 224, mean: Optional[List[float]] = None, std: Optional[List[float]] = None):
    """
    Constructs Torchvision transforms for different ablation settings.
    Default mean and std correspond to ImageNet stats if not specified.
    """
    if mean is None or std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if mode == "none":
        # Raw resize only
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    elif mode == "basic":
        # Flips only
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    elif mode == "strong":
        # Domain-informed botanical augmentation:
        # Leaves have no orientation -> full 180 rotation
        # Subtle color jitter that preserves biological leaf coloration (small hue jitter <= 0.03)
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=180),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.20, scale=(0.02, 0.20), value="random")
        ])
    elif mode == "eval":
        # Deterministic evaluation transform
        return T.Compose([
            T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
    else:
        raise ValueError(f"Unknown transform mode: {mode}")


class MedicinalLeafDataset(Dataset):
    """PyTorch Dataset for Medicinal Leaves."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].tolist()
        self.class_names = df["class_name"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        fp = self.filepaths[idx]
        label = self.labels[idx]
        class_name = self.class_names[idx]

        with Image.open(fp) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)

        return tensor, label, fp


class PreloadedMedicinalLeafDataset(Dataset):
    """
    Fast In-Memory PyTorch Dataset for Medicinal Leaves.
    Preloads and caches resized PIL images in RAM to eliminate disk I/O bottlenecks during CPU training.
    """
    def __init__(self, df: pd.DataFrame, transform=None, preload_size: Optional[Tuple[int, int]] = (224, 224)):
        self.filepaths = df["filepath"].tolist()
        self.labels = df["label"].tolist()
        self.class_names = df["class_name"].tolist()
        self.transform = transform
        self.images = []

        for fp in self.filepaths:
            with Image.open(fp) as img:
                img_rgb = img.convert("RGB")
                if preload_size is not None:
                    img_rgb = img_rgb.resize(preload_size, Image.BILINEAR)
                self.images.append(img_rgb)

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img = self.images[idx]
        label = self.labels[idx]
        fp = self.filepaths[idx]

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            tensor = T.ToTensor()(img)

        return tensor, label, fp



def mixup_cutmix_collate_fn(batch, num_classes=30, mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5):
    """
    Applies MixUp or CutMix to a batch of images and creates smooth one-hot targets.
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    fps = [item[2] for item in batch]

    batch_size = imgs.size(0)
    one_hot_targets = torch.zeros(batch_size, num_classes, dtype=torch.float32)
    one_hot_targets.scatter_(1, labels.unsqueeze(1), 1.0)

    # Label smoothing 0.1
    one_hot_targets = one_hot_targets * 0.9 + (0.1 / num_classes)

    if random.random() > prob:
        return imgs, one_hot_targets, labels, fps

    lam = 1.0
    indices = torch.randperm(batch_size)

    if random.random() < 0.5:
        # MixUp
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        mixed_imgs = lam * imgs + (1 - lam) * imgs[indices]
        mixed_targets = lam * one_hot_targets + (1 - lam) * one_hot_targets[indices]
    else:
        # CutMix
        lam = np.random.beta(cutmix_alpha, cutmix_alpha) if cutmix_alpha > 0 else 1.0
        _, _, H, W = imgs.size()
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cut_w = int(W * np.sqrt(1 - lam))
        cut_h = int(H * np.sqrt(1 - lam))
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        mixed_imgs = imgs.clone()
        mixed_imgs[:, :, y1:y2, x1:x2] = imgs[indices, :, y1:y2, x1:x2]
        adjust_lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        mixed_targets = adjust_lam * one_hot_targets + (1 - adjust_lam) * one_hot_targets[indices]

    return mixed_imgs, mixed_targets, labels, fps
