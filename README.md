# Sistema de Reconhecimento de Gatos da UFMS - Cidade Universitária

Um projeto de Visão Computacional utilizando Deep Learning com PyTorch para identificar gatos específicos presentes no campus da Universidade Federal de Mato Grosso do Sul (UFMS), Campus Campo Grande - Cidade Universitária.

## Sumário
1.  [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2.  [Estrutura do Projeto](#2-estrutura-do-projeto)
3.  [Pré-requisitos e Instalação](#3-pré-requisitos-e-instalação)
4.  [Configuração do Dataset](#4-configuração-do-dataset)
5.  [Algoritmos e Abordagens Teóricas](#5-algoritmos-e-abordagens-teóricas)
    * [Tipo de Aprendizado: Supervisionado](#tipo-de-aprendizado-supervisionado)
    * [Tipo de Problema: Classificação Multi-Classe](#tipo-de-problema-classificação-multi-classe)
    * [Arquitetura do Modelo: Redes Neurais Convolucionais (CNNs)](#arquitetura-do-modelo-redes-neurais-convolucionais-cnns)
    * [Estratégias de Dados: Pré-processamento e Aumento de Dados](#estratégias-de-dados-pré-processamento-e-aumento-de-dados)
    * [Otimização do Modelo](#otimização-do-modelo)
    * [Avaliação e Monitoramento](#avaliação-e-monitoramento)
6.  [Como Executar o Projeto](#6-como-executar-o-projeto)
7.  [Interpretando os Resultados](#7-interpretando-os-resultados)
    * [Saída do Treinamento](#saída-do-treinamento)
    * [Gráficos de Desempenho](#gráficos-de-desempenho)
    * [Matriz de Confusão](#matriz-de-confusão)
    * [Relatório de Classificação (Precision, Recall, F1-Score)](#relatório-de-classificação-precision-recall-f1-score)
8.  [Contribuições](#8-contribuições)
9.  [Licença](#9-licença)

---

## 1. Visão Geral do Projeto

Este projeto tem como objetivo desenvolver um sistema de inteligência artificial capaz de reconhecer gatos individuais do campus da UFMS. A abordagem utilizada é o **Aprendizado Supervisionado** com uma **Rede Neural Convolucional (CNN)**, que é a técnica padrão e mais eficaz para problemas de **Classificação Multi-Classe** em imagens.

O pipeline envolve desde o pré-processamento das imagens até o treinamento do modelo, passando por estratégias de aumento de dados e uma avaliação detalhada do desempenho.

## 2. Estrutura do Projeto

A organização dos arquivos e diretórios é crucial para a clareza e manutenção do projeto:
```bash
IA/
├── gatos/                  # Diretório raiz do dataset. Contém subpastas, cada uma nomeada com o nome de um gato.
│   ├── [nome_do_gato_1]/   # Ex: 'hans_kelsen/'
│   │   ├── imagem_1.jpg
│   │   └── ...
│   ├── [nome_do_gato_2]/   # Ex: 'lina/'
│   │   ├── imagem_1.jpg
│   │   └── ...
│   └── ...
├── modelos/                # Diretório para salvar os modelos treinados e métricas de treinamento (criado automaticamente).
├── pre_processamento.py    # Script com funções para preparar as imagens (redimensionamento, denoising).
├── dataset_gatos.py        # Script que define como carregar o dataset para o PyTorch, incluindo transformações e aumento de dados.
├── modelo_gatos.py         # Script que define a arquitetura da Rede Neural Convolucional (CNN).
├── treino_gatos.py         # Script principal para treinar o modelo.
├── avaliar_modelo.py       # Script para realizar a avaliação detalhada do modelo treinado.
├── visualizar_aumento.py   # Script para visualizar exemplos de imagens com aumento de dados.
└── main.py                 # Script de menu interativo para executar as diferentes partes do projeto.
```
## 3. Pré-requisitos e Instalação

Para rodar este projeto, você precisará ter o Python instalado e as bibliotecas listadas abaixo.

### 3.1. Pré-requisitos

* **Python 3.8 ou superior** (recomendado Python 3.9, 3.10 ou 3.11).

### 3.2. Instalação das Bibliotecas

```bash

pip install torch torchvision numpy opencv-python scikit-learn matplotlib seaborn
```

Observação sobre GPU (CUDA):
Se você possui uma GPU NVIDIA e deseja acelerar o treinamento, instale o PyTorch com suporte a CUDA. As instruções variam ligeiramente dependendo do seu sistema operacional e versão de CUDA. Consulte o site oficial do PyTorch para a instalação correta: https://pytorch.org/get-started/locally/

## 4. Configuração do Dataset
O dataset de imagens deve estar localizado na pasta gatos/ na raiz do projeto.

Estrutura: Cada subdiretório dentro de gatos/ representa uma classe de gato. O nome da pasta será o rótulo da classe.
```bash
gatos/
├── nome_do_gato_1/
│   ├── imagem_gato1_1.jpg
│   └── imagem_gato1_2.png
├── nome_do_gato_2/
│   ├── imagem_gato2_1.jpg
│   └── ...
└── ...
```
## 5. Algoritmos e Abordagens Teóricas

Este projeto emprega diversas técnicas de Machine Learning para resolver o problema de classificação de gatos.

### 5.1. Paradigma de Aprendizado: Supervisionado

* **Conceito:** O modelo aprende a mapear entradas (imagens de gatos) para saídas (nomes dos gatos) a partir de dados pré-rotulados.
* **Aplicação:** Como possuímos imagens de gatos já identificadas, o aprendizado supervisionado é a abordagem natural para treinar o classificador.

### 5.2. Tipo de Problema: Classificação Multi-Classe

* **Conceito:** A tarefa é classificar uma imagem em uma de várias categorias distintas (mais de duas).
* **Aplicação:** O objetivo de identificar entre múltiplos gatos específicos do campus configura um problema de classificação multi-classe.

### 5.3. Modelo Central: Redes Neurais Convolucionais (CNNs)

As CNNs são a escolha fundamental para tarefas de visão computacional devido à sua eficácia em processar dados de imagem.

* **Por que CNNs?**
    * **Extração Automática de Features:** Aprendem hierarquicamente as características visuais mais relevantes das imagens.
    * **Eficiência e Robustez:** Compartilham parâmetros (filtros) e são robustas a pequenas variações de posição (invariância à translação).
* **Componentes Principais:**
    * **Camadas Convolucionais (`nn.Conv2d`):** Detectam padrões locais (bordas, texturas).
    * **Funções de Ativação (ReLU):** Introduzem não-linearidade para aprender relações complexas.
    * **Camadas de Pooling (`nn.MaxPool2d`):** Reduzem a dimensionalidade e a complexidade, mantendo as features essenciais.
    * **Camadas Totalmente Conectadas (`nn.Linear`):** Realizam a classificação final baseada nas features extraídas.
    * **Dropout (`nn.Dropout`):** Uma técnica de regularização para prevenir o overfitting.

### 5.4. Estratégias de Dados

A qualidade e a variedade dos dados são aprimoradas através de:

* **Pré-processamento (`pre_processamento.py`):**
    * **Redimensionamento:** Padroniza todas as imagens para 224x224 pixels.
    * **Normalização de Pixels:** Escala os valores de pixel para [0, 1] (`transforms.ToTensor()`).
    * **Remoção de Ruído (Denoise):** Suaviza ruídos para melhorar a clareza das features.
* **Aumento de Dados (Data Augmentation no `dataset_gatos.py`):**
    * **Conceito:** Cria variações sintéticas das imagens de treino (ex: cortes aleatórios, inversões horizontais) dinamicamente, sem salvar arquivos.
    * **Por que é vital?** Combate o overfitting e aumenta a capacidade de generalização do modelo, especialmente com datasets menores.
    * **Técnicas Usadas:** `RandomResizedCrop`, `RandomHorizontalFlip`.

### 5.5. Otimização do Modelo

O treinamento é um processo de otimização que visa minimizar os erros do modelo.

* **Função de Perda (`nn.CrossEntropyLoss`):**
    * **Conceito:** Mede o "erro" entre a previsão do modelo e o rótulo verdadeiro.
    * **Por que utilizada?** É a função padrão e mais adequada para problemas de classificação multi-classe.
* **Otimizador (Adam - `optim.Adam`):**
    * **Conceito:** Algoritmo que ajusta os pesos do modelo para minimizar a perda. Adam é um otimizador adaptativo, ajustando a taxa de aprendizado para cada parâmetro.
    * **Por que utilizado?** Conhecido por sua robustez e bom desempenho geral em Deep Learning.
    * **Taxa de Aprendizado (`LEARNING_RATE`):** Controla o tamanho dos passos nas atualizações de peso.

### 5.6. Avaliação e Monitoramento

A validação rigorosa garante que o modelo generalize bem para dados novos.

* **Divisão Treino/Validação (`random_split` / `StratifiedShuffleSplit`):**
    * **Conceito:** Separa o dataset em um conjunto para treino e outro para avaliação em dados não vistos.
    * **Por que é essencial?** Permite avaliar a generalização do modelo e detectar overfitting. Para datasets menores, a **divisão estratificada** é preferível para manter a proporção de classes nos conjuntos.
* **Métricas de Desempenho:**
    * **Acurácia:** Percentual de previsões corretas.
    * **Perda (Loss):** Indica o ajuste do modelo aos dados.
    * **Matriz de Confusão:** Tabela que visualiza acertos e erros por classe, identificando confusões específicas do modelo.
    * **Relatório de Classificação (Precision, Recall, F1-Score):** Métricas detalhadas por classe, cruciais para datasets desbalanceados, fornecendo uma visão mais completa do desempenho.
## 6. Como Executar o Projeto
Certifique-se de ter seguido as etapas de Pré-requisitos e Instalação e Configuração do Dataset antes de prosseguir.

Para facilitar a execução, utilize o script main.py com um menu interativo:
```bash
# Navegue até a pasta raiz do projeto (IA/)
cd IA/

# Ative seu ambiente virtual (se ainda não estiver ativado)
# No Windows: .\venv\Scripts\activate
# No macOS/Linux: source venv/bin/activate

# Execute o script principal
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
0. Sair
==============================
```
