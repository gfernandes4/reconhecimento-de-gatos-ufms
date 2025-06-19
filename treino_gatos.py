import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from dataset_gatos import GatosDataset
from modelo_gatos import GatosCNN
import os
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit 

# --- Função principal para treinar o modelo ---
def train_model(root_dir, num_epochs, batch_size, learning_rate, num_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Carrega o dataset completo para obter informações de classes e rótulos
    full_dataset_for_split_info = GatosDataset(root_dir=root_dir, is_train=False) 
    num_classes = len(full_dataset_for_split_info.idx_to_class)
    class_names = full_dataset_for_split_info.idx_to_class

    print(f"Número de classes detectadas: {num_classes}")

    # Verifica se o dataset contém imagens
    train_size = int(0.8 * len(full_dataset_for_split_info))
    val_size = len(full_dataset_for_split_info) - train_size
    
    # Divisão estratificada dos índices
    all_labels = full_dataset_for_split_info.labels 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size / len(full_dataset_for_split_info), random_state=42)
    train_indices, val_indices = next(sss.split(np.zeros(len(full_dataset_for_split_info)), all_labels))

    # Instâncias separadas para treino (com augmentation) e validação (sem augmentation)
    base_train_dataset = GatosDataset(root_dir=root_dir, is_train=True, target_size=full_dataset_for_split_info.target_size)
    base_val_dataset = GatosDataset(root_dir=root_dir, is_train=False, target_size=full_dataset_for_split_info.target_size)

    # Cria subconjuntos para treino e validação usando os índices estratificados
    train_dataset = Subset(base_train_dataset, train_indices)
    val_dataset = Subset(base_val_dataset, val_indices)

    print(f"Dataset de Treino: {len(train_dataset)} imagens")
    print(f"Dataset de Validação: {len(val_dataset)} imagens")

    # Verifica se os datasets não estão vazios
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Verifica se os DataLoaders foram criados corretamente
    print("\n--- Configurando o Modelo ---")
    model = GatosCNN(num_classes=num_classes).to(device)
    print(model)

    # Configura o critério de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n--- Iniciando o Treinamento ---")
    best_accuracy = 0.0
    total_start_time = time.time()
    
    # Listas para armazenar perdas e acurácias durante o treinamento e validação
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Treinamento
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Itera sobre o DataLoader de treino
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Acumula a perda e acurácia
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        # Calcula a perda e acurácia média para o epoch de treino
        epoch_loss = running_loss / len(train_dataset)
        epoch_accuracy = correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Validação
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        # Calcula a perda e acurácia média para o epoch de validação
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_accuracy = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        # Exibe os resultados do epoch
        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch+1}/{num_epochs}\n ============ \n"
              f"Treino Perda: {epoch_loss:.4f}, Treino Acurácia: {epoch_accuracy:.4f}, \n"
              f"Validação Perda: {epoch_val_loss:.4f}, Validação Acurácia: {epoch_val_accuracy:.4f}, \n"
              f"Tempo da Época: {epoch_duration:.2f}s")

        # Salva o melhor modelo
        if epoch_val_accuracy > best_accuracy:
            best_accuracy = epoch_val_accuracy
            model_save_path = os.path.join("modelos", "best_gatos_classifier.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"Acuracia melhorou para: {best_accuracy:.4f}")
            
    total_duration = time.time() - total_start_time
    print(f"\n--- Treinamento Concluído! Tempo Total: {total_duration:.2f} segundos ---")

    # Salva métricas para análise posterior
    np.save(os.path.join("modelos", "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join("modelos", "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join("modelos", "train_accuracies.npy"), np.array(train_accuracies))
    np.save(os.path.join("modelos", "val_accuracies.npy"), np.array(val_accuracies))
    print("Métricas de treinamento salvas para plotagem.")

    # Retorna objetos úteis para avaliação posterior
    return model, train_losses, val_losses, train_accuracies, val_accuracies, class_names, val_loader

if __name__ == '__main__':
    ROOT_DIR = 'gatos'
    NUM_EPOCHS = 8
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_WORKERS = 0  # 0 para Windows
    
    # Traina o modelo e salva os resultados
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies, class_names, val_loader_for_eval = \
        train_model(ROOT_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_WORKERS)

    print("\nTreinamento concluído. Para visualizar os resultados e fazer a avaliação detalhada, execute 'python avaliar_modelo.py'")
    print(f"Os dados para avaliação detalhada estão no diretório: {os.path.join(ROOT_DIR)}")
    print(f"O modelo salvo está em: {os.path.join('modelos', 'best_gatos_classifier.pth')}")
