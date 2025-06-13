import os
import cv2
import numpy as np

# --- Importar funções de pré-processamento do seu arquivo ---
# Certifique-se de que 'pre_processamento.py' (ou o nome que você usou)
# está no mesmo diretório ou em um caminho que o Python possa encontrar.
from pre_processamento import resize_image, normalize_pixels, denoise_image

# --- Configurações de Caminho ---
BASE_DATA_DIR = 'gatos'
TARGET_CAT_FOLDER = 'sofia'
PROCESSED_OUTPUT_DIR = './gatos_pre_processados/sofia_pre_processado' # Nova pasta para as imagens processadas

# Tamanho alvo para redimensionamento
IMG_HEIGHT = 224
IMG_WIDTH = 224

def process_and_save_images(source_dir, output_dir, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """
    Percorre um diretório de origem, aplica pré-processamento a todas as imagens
    e salva as imagens processadas em um diretório de saída.
    """
    if not os.path.exists(source_dir):
        print(f"Erro: O diretório de origem '{source_dir}' não foi encontrado.")
        return

    # Cria o diretório de saída se ele não existir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Diretório de saída '{output_dir}' garantido.")

    print(f"Iniciando o pré-processamento das imagens em: {source_dir}")
    processed_count = 0
    skipped_count = 0

    # Percorre todos os arquivos no diretório de origem
    for filename in os.listdir(source_dir):
        # Verifica se o arquivo é uma imagem (você pode adicionar mais extensões se necessário)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(source_dir, filename)
            output_image_path = os.path.join(output_dir, filename)

            print(f"Processando {filename}...")

            # 1. Redimensionar
            # A função resize_image já faz o imread
            resized_img = resize_image(image_path, target_size=target_size)
            if resized_img is None:
                skipped_count += 1
                continue # Pula para a próxima imagem se não conseguiu carregar

            # 2. Denoise (aplicar APÓS redimensionamento, mas ANTES da normalização para 0-1)
            # A função denoise_image espera uint8, o que o resize_image já retorna.
            denoised_img = denoise_image(resized_img)

            # 3. Normalizar Pixels (aplicar APÓS denoise)
            normalized_img = normalize_pixels(denoised_img)

            # --- Salvar a imagem processada ---
            # Para salvar com OpenCV, geralmente convertemos de volta para uint8 e escala 0-255.
            # Modelos de DL esperam float32 [0,1], mas cv2.imwrite prefere uint8 [0,255].
            # Lembre-se que para o treinamento real do modelo, você usará a versão float32 [0,1].
            # Esta conversão é APENAS para salvar no disco para inspeção.
            output_image_to_save = (normalized_img * 255).astype(np.uint8)
            cv2.imwrite(output_image_path, output_image_to_save)
            processed_count += 1
            print(f"  -> Salvo em: {output_image_path}")
        else:
            print(f"Pulando arquivo não-imagem: {filename}")
            skipped_count += 1

    print(f"\n--- Processamento Concluído ---")
    print(f"Imagens processadas: {processed_count}")
    print(f"Arquivos ignorados (não-imagens ou erro): {skipped_count}")
    print(f"As imagens pré-processadas estão em: {output_dir}")

if __name__ == "__main__":
    # Caminho completo para a pasta específica do gato
    source_cat_path = os.path.join(BASE_DATA_DIR, TARGET_CAT_FOLDER)

    # Caminho para a nova pasta de saída específica para este gato
    # É uma boa prática manter a estrutura de pastas das classes dentro da pasta processada
    output_processed_cat_path = os.path.join(PROCESSED_OUTPUT_DIR, TARGET_CAT_FOLDER)

    process_and_save_images(source_cat_path, output_processed_cat_path, 
                            target_size=(IMG_HEIGHT, IMG_WIDTH))

    print("\nLembre-se: As imagens salvas nova pasta estão em uint8 (0-255).")
    print("Para o treinamento da Rede Neural, você deve usar a versão normalizada (float32, 0-1),")
    print("que é gerada e usada 'on-the-fly' pelo ImageDataGenerator, como no script de treinamento anterior.")