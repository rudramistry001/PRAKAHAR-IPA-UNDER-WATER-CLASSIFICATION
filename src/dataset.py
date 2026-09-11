import os
import glob
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image

def build_metadata(data_root):
    data = []
    base_dir = os.path.join(data_root, 'fish_image')
    class_folders = sorted(glob.glob(os.path.join(base_dir, 'fish_*')))
    
    for class_id, class_folder in enumerate(class_folders):
        class_name = os.path.basename(class_folder)
        image_files = glob.glob(os.path.join(class_folder, '*.png'))
        
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # fish_000000009598_05281.png -> ['fish', '000000009598', '05281']
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 3:
                trajectory_id = parts[1]
                fish_id = parts[2]
            else:
                continue
                
            try:
                # Validate image
                with Image.open(img_path) as img:
                    img.verify()
            except (IOError, SyntaxError):
                print(f"Corrupted image found and skipped: {img_path}")
                continue
                
            rel_path = os.path.relpath(img_path, data_root)
            data.append({
                'image_path': rel_path,
                'class_id': class_id,
                'class_name': class_name,
                'trajectory_id': trajectory_id,
                'fish_id': fish_id
            })
            
    return pd.DataFrame(data)

def create_trajectory_split(metadata_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)
    
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    classes = metadata_df['class_id'].unique()
    for c in classes:
        df_c = metadata_df[metadata_df['class_id'] == c]
        unique_trajectories = df_c['trajectory_id'].unique()
        
        if len(unique_trajectories) < 3:
            # Force at least 1 in val and 1 in test if possible
            np.random.shuffle(unique_trajectories)
            if len(unique_trajectories) == 1:
                train_traj = unique_trajectories
                val_traj = []
                test_traj = []
            elif len(unique_trajectories) == 2:
                train_traj = unique_trajectories[:1]
                val_traj = unique_trajectories[1:]
                test_traj = []
            else:
                train_traj = unique_trajectories[:1]
                val_traj = unique_trajectories[1:2]
                test_traj = unique_trajectories[2:]
        else:
            gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
            train_idx, temp_idx = next(gss1.split(df_c, groups=df_c['trajectory_id']))
            
            train_df = df_c.iloc[train_idx]
            temp_df = df_c.iloc[temp_idx]
            
            relative_val_ratio = val_ratio / (val_ratio + test_ratio)
            gss2 = GroupShuffleSplit(n_splits=1, train_size=relative_val_ratio, random_state=seed)
            val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df['trajectory_id']))
            
            val_df = temp_df.iloc[val_idx]
            test_df = temp_df.iloc[test_idx]
            
            train_dfs.append(train_df)
            val_dfs.append(val_df)
            test_dfs.append(test_df)
            continue
            
        train_dfs.append(df_c[df_c['trajectory_id'].isin(train_traj)])
        val_dfs.append(df_c[df_c['trajectory_id'].isin(val_traj)])
        test_dfs.append(df_c[df_c['trajectory_id'].isin(test_traj)])
        
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    val_df = pd.concat(val_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    
    # Assertions
    train_traj = set(train_df['trajectory_id'])
    val_traj = set(val_df['trajectory_id'])
    test_traj = set(test_df['trajectory_id'])
    
    assert train_traj.isdisjoint(val_traj), "Leakage between train and val!"
    assert train_traj.isdisjoint(test_traj), "Leakage between train and test!"
    assert val_traj.isdisjoint(test_traj), "Leakage between val and test!"
    
    return train_df, val_df, test_df

class Fish4KDataset(Dataset):
    def __init__(self, dataframe, data_root, transform=None, use_mask=False):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform
        self.use_mask = use_mask
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.data_root, row['image_path'])
        
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.use_mask:
            mask_path = img_path.replace('fish_image', 'fish_mask')
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                image[mask == 0] = 0
                
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        label = row['class_id']
        return image, label

def get_class_weights(metadata_df):
    class_counts = metadata_df['class_id'].value_counts().sort_index()
    total_samples = len(metadata_df)
    weights = total_samples / (len(class_counts) * class_counts.values)
    return torch.FloatTensor(weights)

def get_balanced_sampler(dataset):
    class_counts = dataset.dataframe['class_id'].value_counts().sort_index().to_dict()
    sample_weights = [1.0 / class_counts[row['class_id']] for _, row in dataset.dataframe.iterrows()]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler
