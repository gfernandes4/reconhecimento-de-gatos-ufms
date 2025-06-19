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
    * [Testando Gatos Desconhecidos](#testando-gatos-desconhecidos)
7.  [Interpretando os Resultados](#7-interpretando-os-resultados)
    * [Saída do Treinamento](#saída-do-treinamento)
    * [Gráficos de Desempenho](#gráficos-de-desempenho)
    * [Matriz de Confusão](#matriz-de-confusão)
    * [Relatório de Classificação (Precision, Recall, F1-Score)](#relatório-de-classificação-precision-recall-f1-score)
---

## 1. Visão Geral do Projeto

Este projeto tem como objetivo desenvolver um sistema de inteligência artificial capaz de reconhecer gatos individuais do campus da UFMS. A abordagem utilizada é o **Aprendizado Supervisionado** com uma **Rede Neural Convolucional (CNN)**, que é a técnica padrão e mais eficaz para problemas de **Classificação Multi-Classe** em imagens.

O pipeline envolve desde o pré-processamento das imagens até o treinamento do modelo, passando por estratégias de aumento de dados e uma avaliação detalhada do desempenho.

## 2. Estrutura do Projeto

A organização dos arquivos e diretórios é crucial para a clareza e manutenção do projeto:
```bash
IA/
├── gatos/                  # Diretório raiz do dataset. Contém subpastas, cada uma nomeada com o nome de um gato (incluindo 'gatos_da_uf/').
│   ├── [nome_do_gato_1]/   # Ex: 'hans_kelsen/'
│   │   ├── imagem_1.jpg
│   │   └── ...
│   ├── gatos_da_uf/        # Subpasta para gatos da UFMS sem nome específico.
│   │   ├── gato_desconhecido_1.jpg
│   │   └── ...
│   └── ...
├── imagens_teste/          # Diretório para imagens que serão usadas para testar se o gato pertence ao dataset ou é desconhecido.
│   ├── imagem_teste_1.jpg
│   └── ...
├── modelos/                # Diretório para salvar os modelos treinados e métricas de treinamento (criado automaticamente).
├── pre_processamento.py    # Script com funções para preparar as imagens (redimensionamento, denoising).
├── dataset_gatos.py        # Script que define como carregar o dataset para o PyTorch, incluindo transformações e aumento de dados.
├── modelo_gatos.py         # Script que define a arquitetura da Rede Neural Convolucional (CNN).
├── treino_gatos.py         # Script principal para treinar o modelo.
├── avaliar_modelo.py       # Script para realizar a avaliação detalhada do modelo treinado.
├── visualizar_aumento.py   # Script para visualizar exemplos de imagens com aumento de dados.
├── testar_gatos_desconhecidos.py # Script para testar imagens de um diretório específico, classificando ou indicando se é desconhecido.
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
Se você possui uma GPU NVIDIA e deseja acelerar o treinamento, instale o PyTorch com suporte a CUDA. 

As instruções variam ligeiramente dependendo do seu sistema operacional e versão de CUDA. Consulte o site oficial do PyTorch para a instalação correta: https://pytorch.org/get-started/locally/

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

## 7. Interpretando os Resultados
Após executar as opções de treinamento e avaliação, você obterá diversas saídas e visualizações importantes que ajudam a entender o desempenho do seu modelo.

### 7.1. Saída do Treinamento
No console, durante o treinamento (Opção 1), você verá o progresso a cada época, semelhante a este formato:
```bash
Epoch X/Y
 ============
Treino Perda: A.AAAA, Treino Acurácia: B.BBBB, 
Validação Perda: C.CCCC, Validação Acurácia: D.DDDD, 
Tempo da Época: E.EEs
Acuracia melhorou para: D.DDDD
```
- **`Epoch X/Y:`** Indica a época atual (`X`) de um total de `Y` épocas configuradas.
- **`Treino Perda:`** O valor da função de perda nos dados de treinamento.
   - **Ideal:** Deve ser baixo e diminuir consistentemente ao longo das épocas.
   - **Significado:** Mostra o quão bem o modelo está aprendendo e ajustando-se aos dados que já viu.
-**`Treino Acurácia:`** A porcentagem de previsões corretas nos dados de treinamento.
   - **Ideal:** Deve ser alta (próxima de 1.0 ou 100%) e aumentar consistentemente.
   - **Significado:** Mostra a performance do modelo nos dados que ele usou para aprender.
- **`Validação Perda`:** O valor da função de perda nos dados de validação (dados que o modelo nunca viu durante o treinamento).
   - **Ideal:** Deve ser baixo e próximo da `Treino Perda`. Deve diminuir inicialmente e pode estabilizar ou até subir se houver overfitting.
   - **Significado:** É uma medida crucial da capacidade do modelo de generalizar para dados novos.
- **`Validação Acurácia:`** A porcentagem de previsões corretas nos dados de validação.
   - **Ideal:** Deve ser alta (próxima de 1.0 ou 100%) e próxima da `Treino Acurácia`. Deve aumentar inicialmente e estabilizar ou cair se houver overfitting.
   - **Significado:** É a métrica mais importante para avaliar o desempenho do seu modelo no "mundo real".
- **`Tempo da Época:`** O tempo que levou para completar uma época de treinamento e validação.

### Sinal de Alerta: Overfitting
Se a `Treino Acurácia` for muito alta e a `Treino Perda` muito baixa, mas a `Validação Acurácia` for significativamente menor e/ou a `Validação Perda` for muito maior, isso indica **overfitting**. O modelo memorizou os dados de treino e perdeu a capacidade de generalizar.

### 7.2. Gráficos de Desempenho
Após a avaliação (Opção 2), serão gerados gráficos de linha que visualizam a evolução das métricas ao longo das épocas.

 - **Curva de Perda (Loss Curve):** Mostra a `Treino Perda` e `Validação Perda` por época.
   - **Ideal:** Ambas as curvas devem diminuir. A `Validação Perda` deve seguir de perto a `Treino Perda`.
   - **Overfitting**: A `Treino Perd`a continua caindo, mas a `Validação Perda` começa a subir.
- **Curva de Acurácia (Accuracy Curve):** Mostra a `Treino Acurácia` e `Validação Acurácia` por época.
   - **Ideal:** Ambas as curvas devem subir. A `Validação Acurácia` deve seguir de perto a `Treino Acurácia`.
   - **Overfitting:** A `Treino Acuráci` continua subindo, mas a `Validação Acurácia` se estabiliza ou cai.

### 7.3. Matriz de Confusão
Gerada pelo `avaliar_modelo.py`, a matriz de confusão é uma tabela visual que detalha o desempenho do classificador para cada classe:

- **Eixo Y ("Verdadeiro"):** A classe real à qual a imagem pertence.
- **Eixo X ("Previsão"):** A classe que o modelo previu para a imagem.
- **Diagonal Principal (valores mais altos e escuros):** Representa as classificações corretas. Onde a previsão corresponde ao valor verdadeiro.
- **Células Fora da Diagonal Principal:** Representam as classificações incorretas (erros).
   - Um valor na linha "Gato A" e coluna "Gato B" significa que `N` imagens do Gato A real foram erroneamente previstas como Gato B. Isso ajuda a identificar quais gatos o modelo está confundindo entre si.
- **Linhas/Colunas Vazias:** Se uma linha (ou coluna) está completamente vazia (todos zeros), isso pode indicar que não havia nenhuma imagem daquela classe no conjunto de validação. A Divisão Estratificada ajuda a minimizar este problema em datasets pequenos, garantindo que todas as classes estejam representadas.

### 7.4. Relatório de Classificação (Precision, Recall, F1-Score)
Este relatório, também gerado pelo `avaliar_modelo.py`, fornece métricas detalhadas por classe, que são mais informativas que a acurácia geral, especialmente em datasets com classes desbalanceadas:

- `precision` **(Precisão):** Para uma dada classe, é a proporção de previsões positivas corretas em relação ao total de previsões positivas feitas para aquela classe.
   - **Ideal:** Alto. Significa que, quando o modelo diz que é um gato específico, ele geralmente está certo (poucos falsos positivos).
- `recall` **(Sensibilidade):** Para uma dada classe, é a proporção de instâncias positivas reais que foram corretamente identificadas pelo modelo.
   - **Ideal:** Alto. Significa que o modelo consegue encontrar a maioria dos gatos daquela classe quando eles estão presentes (poucos falsos negativos).
- `f1-score:` É a média harmônica (equilibrada) entre Precision e Recall.
   - **Ideal:** Alto. É uma métrica útil quando você precisa de um bom equilíbrio entre Precisão e Recall.
- `support:` O número real de imagens de cada classe no conjunto de validação/teste.

Essas métricas são cruciais para ter uma visão completa do desempenho do seu modelo, identificando não apenas a taxa geral de acertos, mas também onde ele está acertando e errando para cada categoria de gato.
