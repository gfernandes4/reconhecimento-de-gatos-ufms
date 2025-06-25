import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset_gatos import GatosDataset # Para obter os nomes das classes
from modelo_gatos import GatosCNN
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np # Necessário para np.concatenate

# --- Configurações para carregar o modelo e dados ---
ROOT_DIR = 'gatos' # Precisa ser o mesmo diretório usado no treino
MODEL_PATH = os.path.join("modelos", "best_gatos_classifier.pth")
BATCH_SIZE = 32 # O mesmo batch_size ou um múltiplo dele
NUM_WORKERS = 0 

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo para avaliação: {device}")

    # --- 1. Preparar o Dataset e DataLoader para Avaliação ---
    # Precisamos do dataset completo para obter os nomes das classes e criar o val_loader
    print("\n--- Preparando dados para avaliação ---")
    full_dataset = GatosDataset(root_dir=ROOT_DIR)
    num_classes = len(full_dataset.idx_to_class)
    class_names = full_dataset.idx_to_class # Nomes das classes para os gráficos e relatório

    # Para a avaliação, usaremos o conjunto de validação novamente
    # É importante recriar a divisão para garantir consistência ou carregar o estado
    # do random_split. Para simplicidade, vamos recriar a divisão aqui.
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    # Apenas o val_dataset é relevante para a avaliação final neste script
    _, val_dataset = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"Dataset de Validação/Teste para avaliação: {len(val_dataset)} imagens")

    # --- 2. Carregar o Modelo Treinado ---
    print("\n--- Carregando o modelo treinado ---")
    model = GatosCNN(num_classes=num_classes).to(device)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Melhor modelo carregado de: {MODEL_PATH}")
    else:
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}. Certifique-se de que o treinamento foi executado e salvou o modelo.")
        exit() # Sai se o modelo não for encontrado

    model.eval() # Coloca o modelo em modo de avaliação

    # --- 3. Coletar Previsões e Rótulos para Avaliação Detalhada ---
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- 4. Gerar e Visualizar Métricas de Desempenho ---
    print("\n--- Gerando Matriz de Confusão ---")
    
    # Matriz de Confusão
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatriz de Confusão:")
    print(cm)

    plt.figure(figsize=(10, 8)) # Aumenta um pouco o tamanho para mais classes
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Previsão')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(os.path.join("modelos", "confusion_matrix.png"))
    plt.show()
    plt.close()

    # Relatório de Classificação (Precision, Recall, F1-Score)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    