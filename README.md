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

É altamente recomendável criar um ambiente virtual para gerenciar as dependências do projeto.

```bash
# 1. Navegue até a pasta raiz do seu projeto (IA/)
cd IA/

# 2. Crie um ambiente virtual (se ainda não tiver um)
python -m venv venv

# 3. Ative o ambiente virtual
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

# 4. Instale as bibliotecas necessárias
pip install torch torchvision numpy opencv-python scikit-learn matplotlib seaborn
```
