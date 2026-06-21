# Reconhecimento de Imagens com CNN e Transfer Learning

Projeto desenvolvido na disciplina de Inteligencia Artificial da Unisinos, com orientacao do professor Gabriel de Oliveira Ramos. O trabalho implementa um pipeline completo de visao computacional supervisionada para classificacao multiclasse de imagens de bolas esportivas.

O objetivo foi comparar duas abordagens:

- uma CNN treinada do zero, baseada em SE-ResNet-34
- uma abordagem com Transfer Learning usando EfficientNetB0 pre-treinada no ImageNet

O projeto cobre desde a preparacao dos dados ate a avaliacao final dos modelos, incluindo pre-processamento, augmentacao, arquitetura, treinamento, validacao, matriz de confusao e metricas por classe.

## Integrantes

- Arthur Schallenberger
- Giovani de Souza
- Leonardo Fronza
- Renan Milech Pereira

## Contexto

Muitas das tecnologias usadas diariamente em IA moderna funcionam "por baixo dos panos". Por isso, este trabalho foi pensado para ir alem do uso de modelos prontos: a proposta foi entender a logica, a matematica e os desafios praticos por tras de um sistema real de reconhecimento de imagens.

O resultado foi a construcao de uma estrutura completa de aprendizado supervisionado, comparando uma rede treinada integralmente do zero com uma estrategia baseada em pesos pre-treinados.

## Problema

O problema abordado e a classificacao multiclasse de imagens de bolas esportivas. Dada uma imagem colorida, o modelo deve identificar corretamente a classe da bola entre 15 categorias diferentes.

Esse e um problema de visao computacional supervisionada com rotulo unico por imagem. O principal desafio esta na semelhanca visual entre algumas classes, como:

- football e rugby_ball
- hockey_ball e hockey_puck
- cricket_ball e tennis_ball

Enquanto algumas classes possuem padroes bem distintos, outras exigem que o modelo aprenda texturas, formas e contextos visuais com maior precisao.

## Dataset

Foi utilizado o dataset Sports Ball Image Recognition, disponibilizado publicamente no Kaggle.

- 15 classes
- 7.328 imagens de treino
- 1.841 imagens de teste
- 9.169 imagens no total
- formato JPEG
- imagens RGB
- resolucao original variada
- redimensionamento para 224x224 em todos os experimentos

As classes avaliadas sao:

- american_football
- baseball
- basketball
- billiard_ball
- bowling_ball
- cricket_ball
- football
- golf_ball
- hockey_ball
- hockey_puck
- rugby_ball
- shuttlecock
- table_tennis_ball
- tennis_ball
- volleyball

O conjunto apresenta leve desbalanceamento entre classes, mas dentro de uma faixa ainda administravel para o problema.

## Pipeline do Projeto

O notebook implementa todo o fluxo da solucao:

1. deteccao e organizacao do dataset
2. separacao entre treino, validacao e teste
3. criacao de pipelines com `tf.data`
4. redimensionamento e normalizacao das imagens
5. data augmentation para aumentar robustez
6. treinamento de uma CNN propria
7. treinamento de um modelo com transfer learning
8. avaliacao com acuracia, loss, matriz de confusao e classification report

## Pre-processamento e Augmentacao

As imagens foram padronizadas para 224x224 pixels, tamanho adotado tanto pela CNN propria quanto pela EfficientNetB0.

Foram utilizadas tecnicas de augmentacao apenas no treino, incluindo:

- flip horizontal e vertical
- rotacao em multiplos de 90 graus
- translacao
- zoom
- color jitter
- cutout

Essas transformacoes foram importantes para melhorar a generalizacao, especialmente porque o dataset possui grande variacao de iluminacao, enquadramento, escala e fundo.

## Modelos Avaliados

## 1. CNN propria: SE-ResNet-34

A principal arquitetura treinada do zero foi uma SE-ResNet-34, com:

- blocos residuais
- atencao de canal com Squeeze-and-Excitation
- GlobalAveragePooling2D
- regularizacao com dropout
- treinamento com AdamW
- cosine annealing com warmup
- label smoothing

Essa abordagem foi usada como baseline forte sem qualquer conhecimento pre-treinado.

## 2. Transfer Learning: EfficientNetB0

Na segunda abordagem, foi utilizada a EfficientNetB0 pre-treinada no ImageNet.

O treinamento foi dividido em duas fases:

1. treinamento apenas da cabeca de classificacao com a base congelada
2. fine-tuning das ultimas camadas da base convolucional

Essa estrategia permite aproveitar representacoes ja aprendidas em larga escala e adaptalas ao dominio especifico do dataset.

## Resultados

Comparativo final dos experimentos:

| Modelo | Estrategia | Epocas | Acuracia de Teste | Loss de Teste |
|---|---|---:|---:|---:|
| SE-ResNet-34 | Treinamento do zero | 120 | 75,77% | 1,4057 |
| EfficientNetB0 | Transfer Learning + fine-tuning | 30 + 50 | 87,89% | 0,4463 |

O uso de transfer learning trouxe ganho de 12,12 pontos percentuais sobre o modelo treinado do zero.

### Principais observacoes

- A SE-ResNet-34 entregou um resultado forte mesmo sem pesos pre-treinados.
- A EfficientNetB0 teve melhor desempenho geral e convergiu com mais eficiencia.
- Classes visualmente simples ou parecidas foram as mais desafiadoras, como hockey_puck, bowling_ball, hockey_ball e american_football.
- O pipeline com `tf.data` ajudou no desempenho e na reproducibilidade dos experimentos.

## Estrutura do Repositorio

Atualmente o repositorio contem:

- `Projeto.ipynb`: notebook principal com toda a implementacao e analise
- `DescricaoDoTrabalho.pdf`: descricao do trabalho proposto

## Tecnologias Utilizadas

- Python
- TensorFlow / Keras
- scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Kaggle API
- Jupyter Notebook

## Como Executar

O projeto foi estruturado principalmente para execucao em notebook, com suporte a ambiente Colab e uso do dataset do Kaggle.

Passos gerais:

1. obter o dataset `samuelcortinhas/sports-balls-multiclass-image-classification`
2. garantir a estrutura com pastas `train/` e `test/`
3. instalar as dependencias Python necessarias
4. abrir o notebook `Projeto.ipynb`
5. executar as celulas na ordem

Em ambiente Colab, o notebook tambem contem logica para:

- configurar credenciais da Kaggle API
- baixar o dataset automaticamente
- salvar artefatos de modelo e logs

