import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image


class FoodDataset(Dataset):
    def __init__(self, df, ingr2idx, all_ingredients, mass_mean, mass_std, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.ingr2idx = ingr2idx
        self.all_ingredients = all_ingredients
        self.mass_mean = mass_mean
        self.mass_std = mass_std
        self.image_dir = Path(image_dir)
        self.transform = transform

    def encode_ingredients(self, ingredients_str):
        vector = np.zeros(len(self.all_ingredients), dtype=np.float32)

        for ingr in ingredients_str.split(';'):
            ingr = ingr.strip()
            if ingr in self.ingr2idx:
                vector[self.ingr2idx[ingr]] = 1.0

        return vector

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image
        img_path = self.image_dir / row['dish_id'] / 'rgb.png'
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        # tabular features
        ingr_vector = self.encode_ingredients(row['ingredients'])
        mass = (row['total_mass'] - self.mass_mean) / self.mass_std
        tabular_features = np.concatenate([ingr_vector, [mass]]).astype(np.float32)

        target = np.float32(row['total_calories'])

        return (
            image,
            torch.tensor(tabular_features, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32)
        )