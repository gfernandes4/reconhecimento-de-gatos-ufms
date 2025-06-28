# 🐾 Sistema de Reconhecimento de Gatos da UFMS - Cidade Universitária 🐾

Este projeto de Visão Computacional, usando **Deep Learning com PyTorch**, tem o objetivo de **identificar gatos específicos** que habitam o campus da Universidade Federal de Mato Grosso do Sul (UFMS), Campo Grande - Cidade Universitária.

## Sumário
1.  [Estrutura do Projeto](#2-estrutura-do-projeto)
2.  [Pré-requisitos e Instalação](#3-pré-requisitos-e-instalação)
3.  [Como Executar o Projeto](#6-como-executar-o-projeto)

---

## 1. Estrutura do Projeto

```bash
IA/
├── gatos/                  # Dataset de imagens, com subpastas para cada classe de gato (e.g., 'hans_kelsen/', 'gatos_da_uf/').
├── imagens_teste/          # Imagens para testar o reconhecimento (gatos conhecidos vs. desconhecidos).
├── modelos/                # Salva modelos treinados e métricas (matrizes de confusão, gráficos de desempenho).
├── pre_processamento.py    # Funções para preparar imagens (redimensionamento, denoising).
├── dataset_gatos.py        # Define como carregar o dataset para PyTorch, incluindo transformações e aumento de dados.
├── modelo_gatos.py         # Define a arquitetura da Rede Neural Convolucional (CNN).
├── treino_gatos.py         # Script principal para treinar o modelo.
├── avaliar_modelo.py       # Avaliação detalhada do modelo treinado.
├── visualizar_aumento.py   # Visualiza exemplos de imagens com aumento de dados.
├── testar_gatos_desconhecidos.py # Classifica gatos de 'imagens_teste/', identificando desconhecidos.
└── main.py                 # Menu interativo para executar as diferentes partes do projeto.
```
## 2. Pré-requisitos e Instalação

Para rodar este projeto, você precisará ter o Python instalado e as bibliotecas listadas abaixo.

### 2.1. Pré-requisitos

* **Python 3.8 ou superior** (recomendado Python 3.9, 3.10 ou 3.11).

### 2.2. Instalação das Bibliotecas

```bash

pip install torch torchvision numpy opencv-python scikit-learn matplotlib seaborn
```

Observação sobre GPU (CUDA):
Se você possui uma GPU NVIDIA e deseja acelerar o treinamento, instale o PyTorch com suporte a CUDA. 

As instruções variam ligeiramente dependendo do seu sistema operacional e versão de CUDA. Consulte o site oficial do PyTorch para a instalação correta: https://pytorch.org/get-started/locally/


## 3. Como Executar o Projeto
Certifique-se de ter seguido as etapas de Pré-requisitos e Instalação e Configuração do Dataset antes de prosseguir.

Para facilitar a execução, utilize o script main.py com um menu interativo:
```bash
python main.py
```
Você verá um menu com as seguintes opções:
```bash
==============================
Menu Principal do Projeto Gatos IA
==============================
1. Treinar Modelo (treino_gatos.py)
2. Avaliar Modelo (avaliar_modelo.py)
3. Visualizar Aumento de Dados (visualizar_aumento.py)
4. Testar Gatos Desconhecidos (testar_gatos_desconhecidos.py)
0. Sair
==============================
```
## Testando Gatos Desconhecidos
Ao escolher a opção 4. Testar Gatos Desconhecidos, o script testar_gatos_desconhecidos.py será executado automaticamente.

- Ele irá procurar por imagens dentro do diretório imagens_teste/ (localizado na raiz do projeto).
- Para cada imagem, o modelo tentará classificá-la como um gato conhecido (com nome), um "gato da UF" (sem nome específico), ou indicará que é um gato desconhecido/fora do dataset se a confiança da previsão for abaixo de um limiar (CONFIDENCE_THRESHOLD).
- Os resultados serão exibidos no console e visualmente em uma janela pop-up para cada imagem.


