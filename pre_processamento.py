import cv2
import numpy as np
import os

# Redimensionamento
def resize_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro: Não foi possível carregar a imagem em {image_path}")
        return None
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img_resized

# Normalização de Pixels
def normalize_pixels(image_array):
    normalized_img = image_array.astype(np.float32) / 255.0
    return normalized_img

# Denoise (remoção de ruído)
def denoise_image(image_array):
    if image_array.dtype != np.uint8:
        print("Aviso: Denoise espera imagem uint8. Convertendo para uint8 para aplicar o filtro.")
        image_array = cv2.convertScaleAbs(image_array)
    denoised_img = cv2.medianBlur(image_array, 5)
    return denoised_img

if __name__ == "__main__":
    # Caminho da imagem de teste
    image_folder = "gatos/hans_kelsen"
    image_name = "hans_kelsen_dormindo.jpg"
    image_path = os.path.join(image_folder, image_name)

    if not os.path.exists(image_folder):
        print(f"A pasta '{image_folder}' não existe. Por favor, crie-a e coloque uma imagem de teste.")
        exit()

    print(f"Testando pré-processamento na imagem: {image_path}")

    print("\nPasso 1: Redimensionando a imagem...")
    resized_img = resize_image(image_path, target_size=(224, 224))
    if resized_img is None:
        exit()

    print("Passo 2: Removendo ruído da imagem redimensionada...")
    denoised_img = denoise_image(resized_img)

    print("Passo 3: Normalizando os pixels da imagem denoised para 0-1...")
    normalized_img = normalize_pixels(denoised_img)

    # Visualização das Imagens
    original_img = cv2.imread(image_path)

    print("\nExibindo imagens... Feche cada janela para ver a próxima.")

    if original_img is not None:
        cv2.imshow("1. Original Image", original_img)
        print("  - Exibindo Original (pressione qualquer tecla para continuar)...")
        cv2.waitKey(0)

    cv2.imshow("2. Resized Image (224x224)", resized_img)
    print("  - Exibindo Redimensionada (pressione qualquer tecla para continuar)...")
    print(f"    Dimensões da imagem redimensionada: {resized_img.shape}")
    print(f"    Tipo de dado da imagem redimensionada: {resized_img.dtype}")
    cv2.waitKey(0)

    cv2.imshow("3. Denoised Image (Median Blur)", denoised_img)
    print("  - Exibindo Denoised (pressione qualquer tecla para continuar)...")
    print(f"    Tipo de dado da imagem denoised: {denoised_img.dtype}")
    cv2.waitKey(0)

    # Para exibir imagem normalizada, cv2.imshow aceita float32 [0,1]
    cv2.imshow("4. Normalized Image (0-1)", normalized_img)
    print("  - Exibindo Normalizada (pressione qualquer tecla para continuar)...")
    print(f"    Tipo de dado da imagem normalizada: {normalized_img.dtype}")
    print(f"    Valor mínimo de pixel: {np.min(normalized_img):.4f}, Valor máximo de pixel: {np.max(normalized_img):.4f}")
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\nTeste de pré-processamento concluído!")
