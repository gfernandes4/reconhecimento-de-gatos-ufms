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

# função para treinar o modelo por uma época e validar 
def train_single_epoch(model, train_loader, criterion, optimizer, device):
  
    model.train()
    running_loss_epoch = 0.0
    correct_train_epoch = 0
    total_train_epoch = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss_epoch += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_train_epoch += labels.size(0)
        correct_train_epoch += (predicted == labels).sum().item()

    epoch_loss = running_loss_epoch / len(train_loader.dataset)
    epoch_accuracy = correct_train_epoch / total_train_epoch
    return epoch_loss, epoch_accuracy

def validate_single_epoch(model, val_loader, criterion, device):
    
    model.eval()
    val_loss_epoch = 0.0 # Mantemos este cálculo para o gráfico de validação
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss_epoch += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    epoch_val_loss = val_loss_epoch / len(val_loader.dataset) # Perda de validação
    epoch_val_accuracy = correct_val / total_val # Acurácia de validação
    return epoch_val_loss, epoch_val_accuracy # Retorna ambos, mas só um será impresso

# Função principal para treinar o modelo 
def train_model(root_dir, num_epochs, batch_size, learning_rate, num_workers=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}") 

    # Carrega o dataset completo para obter informações de classes e rótulos
    full_dataset_for_split_info = GatosDataset(root_dir=root_dir, is_train=False) 
    num_classes = len(full_dataset_for_split_info.idx_to_class)
    class_names = full_dataset_for_split_info.idx_to_class

    print(f"Número de classes detectadas: {num_classes}") 
    
    # Verifica se o número de classes é maior que 1
    train_size = int(0.8 * len(full_dataset_for_split_info))
    val_size = len(full_dataset_for_split_info) - train_size
    
    # Realiza a divisão estratificada para garantir que as classes sejam balanceadas
    all_labels = full_dataset_for_split_info.labels 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size / len(full_dataset_for_split_info), random_state=42)
    train_indices, val_indices = next(sss.split(np.zeros(len(full_dataset_for_split_info)), all_labels))

    # Cria os datasets de treino e validação usando os índices obtidos
    base_train_dataset = GatosDataset(root_dir=root_dir, is_train=True, target_size=full_dataset_for_split_info.target_size)
    base_val_dataset = GatosDataset(root_dir=root_dir, is_train=False, target_size=full_dataset_for_split_info.target_size)

    # Usa os índices para criar subconjuntos
    train_dataset = Subset(base_train_dataset, train_indices)
    val_dataset = Subset(base_val_dataset, val_indices)

    print(f"Dataset de Treino: {len(train_dataset)} imagens") 
    print(f"Dataset de Validação: {len(val_dataset)} imagens") 

    # Cria DataLoaders para treino e validação
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("\n--- Configurando o Modelo ---") 
    model = GatosCNN(num_classes=num_classes).to(device)
    print(model) # Mantém este print

    # Configura o critério de perda e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print("\n--- Iniciando o Treinamento ---") 
    best_accuracy = 0.0
    total_start_time = time.time()
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        epoch_loss, epoch_accuracy = train_single_epoch(model, train_loader, criterion, optimizer, device)
        epoch_val_loss, epoch_val_accuracy = validate_single_epoch(model, val_loader, criterion, device)
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        val_losses.append(epoch_val_loss) 
        val_accuracies.append(epoch_val_accuracy)

        # Imprime os resultados no final da época
        epoch_duration = time.time() - epoch_start_time
        print(f"\nÉpoca {epoch+1}/{num_epochs} Concluída:")
        print(f"  Perda de Treino: {epoch_loss:.4f}")
        print(f"  Acurácia de Treino: {epoch_accuracy:.4f}")
        print(f"  Acurácia de Validação: {epoch_val_accuracy:.4f}")
        print(f"  Tempo da Época: {epoch_duration:.2f}s")

        # Salva o melhor modelo
        if epoch_val_accuracy > best_accuracy:
            best_accuracy = epoch_val_accuracy
            model_save_path = os.path.join("modelos", "best_gatos_classifier.pth")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  Acurácia de validação melhorou para: {best_accuracy:.4f} (Modelo Salvo)")

    total_duration = time.time() - total_start_time
    print(f"\n--- Treinamento Concluído! Tempo Total: {total_duration:.2f} segundos ---") 

    np.save(os.path.join("modelos", "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join("modelos", "val_losses.npy"), np.array(val_losses))
    np.save(os.path.join("modelos", "train_accuracies.npy"), np.array(train_accuracies))
    np.save(os.path.join("modelos", "val_accuracies.npy"), np.array(val_accuracies))
    print("Métricas de treinamento salvas para plotagem.")

    total_samples_processed_train = len(train_dataset) * num_epochs
    total_samples_processed_val = len(val_dataset) * num_epochs
    
    print(f"\n--- Resumo do Processamento ---") 
    print(f"Número total de amostras (imagens originais) no conjunto de TREINO: {len(train_dataset)}")
    print(f"Número total de amostras (imagens originais) no conjunto de VALIDAÇÃO: {len(val_dataset)}")
    print(f"Modelo treinado por: {num_epochs} épocas")
    print(f"Número total de amostras processadas no TREINO (com aumento): {total_samples_processed_train}")
    print(f"Número total de amostras processadas na VALIDAÇÃO: {total_samples_processed_val}")

    return model, train_losses, val_losses, train_accuracies, val_accuracies, class_names, val_loader

if __name__ == '__main__':
    ROOT_DIR = 'gatos'
    NUM_EPOCHS = 11
    BATCH_SIZE = 32 
    LEARNING_RATE = 0.001 # Taxa de aprendizado (pesos)
    NUM_WORKERS = 0 
    
    # Traina o modelo e salva os resultados
    trained_model, train_losses, val_losses, train_accuracies, val_accuracies, class_names, val_loader_for_eval = \
        train_model(ROOT_DIR, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, NUM_WORKERS)

    print("\nTreinamento concluído.")
