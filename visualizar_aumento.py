import torch
from torch.utils.data import DataLoader
from dataset_gatos import GatosDataset
import matplotlib.pyplot as plt
import torchvision

# Função para exibir uma imagem usando matplotlib
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))  # Transforma tensor para formato de imagem (H, W, C)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pausa para garantir que a imagem seja exibida

ROOT_DIR = 'gatos'  # Diretório onde estão as imagens dos gatos
BATCH_SIZE = 8      # Tamanho do batch para visualização
NUM_WORKERS = 0     # Número de workers para o DataLoader

if __name__ == '__main__':
    print("Preparando dataset para visualização do aumento de dados...")
    # Cria o dataset com possíveis aumentos de dados (data augmentation)
    temp_dataset_for_viz = GatosDataset(root_dir=ROOT_DIR, is_train=True, target_size=(224, 224))
    if len(temp_dataset_for_viz) == 0:
        print(f"Erro: Nenhum gato encontrado no diretório {ROOT_DIR}.")
        exit()
    # Cria o DataLoader para carregar os dados em batches
    temp_loader_for_viz = DataLoader(temp_dataset_for_viz, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"Obtendo um batch de {BATCH_SIZE} imagens aumentadas para visualização...")
    try:
        # Obtém um batch de imagens e seus rótulos
        inputs, labels = next(iter(temp_loader_for_viz))
    except Exception as e:
        print(f"Erro ao carregar o primeiro batch. Erro: {e}")
        exit()
    # Junta as imagens do batch em uma grade para visualização
    out = torchvision.utils.make_grid(inputs)
    # Obtém os nomes das classes correspondentes aos rótulos
    class_names_for_labels = [temp_dataset_for_viz.idx_to_class[label.item()] for label in labels]
    plt.figure(figsize=(15, 15))
    # Exibe a grade de imagens com os nomes das classes
    imshow(out, title=class_names_for_labels)
    plt.show()
    print("\nVisualização concluída.")
