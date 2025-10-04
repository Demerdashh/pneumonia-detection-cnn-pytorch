import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.image_paths, self.labels = self._load_images()

    def _load_images(self):
        image_paths, labels = [], []
        for label, idx in LABEL_MAP.items():
            folder = os.path.join(self.root_dir, label)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(folder, fname))
                    labels.append(idx)
        return image_paths, labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
