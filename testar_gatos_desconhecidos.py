import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from modelo_gatos import GatosCNN
from dataset_gatos import GatosDataset
from pre_processamento import preprocess_image_for_model

# Configurações principais
CONFIDENCE_THRESHOLD = 0.70  # Probabilidade mínima para considerar identificação válida
TARGET_IMG_SIZE = (224, 224) # Tamanho de entrada esperado pelo modelo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("modelos", "best_gatos_classifier.pth")

# Pasta com imagens para teste automático
TEST_IMAGES_DIR = "imagens_teste" 

# Carrega modelo treinado e nomes das classes
def load_model_and_metadata():
    print(f"Usando dispositivo: {DEVICE}")
    print("Carregando informações do dataset para inicializar o modelo...")
    
    temp_dataset_info = GatosDataset(root_dir='gatos', is_train=False) 
    num_classes = len(temp_dataset_info.idx_to_class) # Número de classes detectadas
    class_names = temp_dataset_info.idx_to_class # Lista de nomes das classes
    
    model = GatosCNN(num_classes=num_classes).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.eval()
            print(f"Modelo carregado de: {MODEL_PATH}")
            return model, class_names
        except Exception as e:
            print(f"Erro ao carregar o modelo de {MODEL_PATH}: {e}. "
                  "Verifique se o modelo foi salvo corretamente e que a arquitetura do GatosCNN não mudou.")
            return None, None
    else:
        print(f"Erro: Arquivo do modelo não encontrado em {MODEL_PATH}. Por favor, treine o modelo primeiro!")
        return None, None

# Classifica uma imagem e exibe o resultado com base na confiança
def classify_and_display_image(image_path, model, class_names, confidence_threshold):
    
    try:
        processed_image_np = preprocess_image_for_model(image_path, target_size=TARGET_IMG_SIZE)
    except FileNotFoundError as e:
        print(f"Erro ao carregar {image_path}: {e}")
        return

    # Prepara tensor para o modelo
    input_tensor = torch.from_numpy(processed_image_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        max_prob, predicted_idx = torch.max(probabilities, 1)
        
        predicted_class_name = class_names[predicted_idx.item()]
        confidence = max_prob.item()

    status_text = ""
    display_confidence_in_title = True

    # Define mensagem de status conforme classe e confiança
    if predicted_class_name == "gatos_da_uf":
        if confidence >= confidence_threshold:
            status_text = "Gato da UF (não nomeado)"
        else:
            status_text = "Gato da UF (não nomeado, baixa confiança)" 
    else: 
        if confidence >= confidence_threshold:
            status_text = f"Gato identificado: {predicted_class_name}"
        else:
            status_text = f"Não é um gato da UF (desconhecido/fora do dataset)"
            display_confidence_in_title = False 

    print(f"\n--- Imagem: {os.path.basename(image_path)} ---")
    print(f"Status: {status_text}")
    if display_confidence_in_title:
        print(f"Confiança na previsão principal: {confidence:.4f}")
    else:
        print(f"Maior confiança ({confidence:.4f}) abaixo do limiar {confidence_threshold:.2f}.")

    # Carrega imagem original para exibição
    original_img_bgr = cv2.imread(image_path)
    if original_img_bgr is None:
        img_bytes = np.fromfile(image_path, np.uint8)
        original_img_bgr = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if original_img_bgr is None:
            print(f"Não foi possível carregar a imagem original para exibição: {image_path}")
            return
    
    original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(original_img_rgb)
    
    title_text_for_plot = status_text
    if display_confidence_in_title:
        title_text_for_plot += f" ({confidence:.2f})"
    
    plt.title(title_text_for_plot, fontsize=14)
    plt.axis('off')
    plt.show()

# Execução principal: percorre imagens da pasta e classifica uma a uma
if __name__ == '__main__':
    model, class_names = load_model_and_metadata()
    if model is None or class_names is None:
        exit() # Sai se não conseguir carregar modelo

    print(f"\nIniciando teste automático das imagens no diretório: '{TEST_IMAGES_DIR}'")
    print(f"Limiar de Confiança configurado: {CONFIDENCE_THRESHOLD:.2f}")

    if not os.path.exists(TEST_IMAGES_DIR) or not os.path.isdir(TEST_IMAGES_DIR):
        print(f"Erro: O diretório '{TEST_IMAGES_DIR}' não foi encontrado. Por favor, crie-o e coloque suas imagens de teste lá.")
        exit()
    
    # Lista arquivos de imagem suportados
    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    if not image_files:
        print(f"Nenhuma imagem suportada encontrada na pasta '{TEST_IMAGES_DIR}'.")
        exit()
    
    print(f"Encontradas {len(image_files)} imagens na pasta. Exibindo uma por uma.")
    for img_file in tqdm(image_files, desc="Classificando Imagens"):
        full_path = os.path.join(TEST_IMAGES_DIR, img_file)
        classify_and_display_image(full_path, model, class_names, CONFIDENCE_THRESHOLD)
        # Aguarda usuário antes de mostrar próxima imagem
        _ = input("Pressione Enter para a próxima imagem ou feche a janela para continuar...")
        
    print("\nTeste de imagens desconhecidas concluído!")
