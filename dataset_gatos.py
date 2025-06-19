import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from torchvision import transforms

class GatosDataset(Dataset):
    def __init__(self, root_dir, is_train=True, target_size=(224, 224)):
        self.root_dir = root_dir
        self.image_paths = []  # Caminhos das imagens
        self.labels = []       # Labels das imagens
        self.class_to_idx = {} # Mapeia nome da classe para índice
        self.idx_to_class = [] # Lista de nomes das classes
        self.target_size = target_size # Tamanho alvo para as imagens

        # Percorre as pastas de classes
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                self.idx_to_class.append(class_name)
                # Adiciona imagens e labels
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(idx)

        # Transformação padrão
        
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        # Transforms diferentes para treino e validação
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(self.target_size[0], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_paths)  # Número de imagens

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = cv2.imread(img_path)  # Lê a imagem
            if img is None:
                # Tenta ler imagens com caminhos especiais
                img_bytes = np.fromfile(img_path, np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem em {img_path}.")
            img = denoise_image(img)  # Remove ruído
            image_tensor = self.transform(img)  # Aplica transformações
        except FileNotFoundError as e:
            print(e)
            raise

        return image_tensor, label  # Retorna imagem e label

def denoise_image(image_array):
    # Garante que a imagem está no formato uint8
    if image_array.dtype != np.uint8:
        if np.max(image_array) <= 1.0 and image_array.dtype == np.float32:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = cv2.convertScaleAbs(image_array)
    denoised_img = cv2.medianBlur(image_array, 5)  # Aplica filtro de mediana
    return denoised_img