import cv2
import numpy as np

def resize_image(image, target_size=(224, 224)):
    
    img_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return img_resized

def normalize_pixels(image_array):
    
    normalized_img = image_array.astype(np.float32) / 255.0
    return normalized_img

def denoise_image(image_array):
   
    # Garante que a imagem está no formato uint8 para o filtro funcionar corretamente
    if image_array.dtype != np.uint8:
        if np.max(image_array) <= 1.0 and image_array.dtype == np.float32:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = cv2.convertScaleAbs(image_array)
    denoised_img = cv2.medianBlur(image_array, 5)
    return denoised_img

def preprocess_image_for_model(image_path, target_size=(224, 224)):
  
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Erro: Não foi possível carregar a imagem em {image_path}.")
    resized_img = resize_image(img, target_size)
    denoised_img = denoise_image(resized_img)
    normalized_img = normalize_pixels(denoised_img)
    return normalized_img
