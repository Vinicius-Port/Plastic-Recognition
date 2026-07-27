# ♻️ Plastic Recognition: Classification & Automated MLOps Pipeline

Este repositório contém a solução completa para classificação automatizada de resíduos plásticos (PET, PE_HD, PP, PS e Outros) utilizando **Visão Computacional e Deep Learning**, estruturada tanto para **pesquisa científica (estudo de benchmark de 8 modelos)** quanto para **operação em produção (Pipeline MLOps automatizado)**.

---

## 📂 Estrutura do Repositório

```
Plastic-Recognition/
│
├── 📂 pipeline/                  # 🚀 PIPELINE AUTOMATIZADO (MLOps Production Code)
│   ├── ingest_dataset.py         # Padronização WaDaBa e Data Quality Gate
│   ├── train_pipeline.py         # Treinamento modular resiliente & auto-saver
│   ├── model_registry.py         # Leaderboard automático & exportação ONNX
│   └── simulate_belt_pipeline.py # Simulador em tempo real com Gabarito (Ground Truth)
│
├── 📂 benchmark/                 # 📊 EXPERIMENTOS DA DISSERTAÇÃO (Benchmark 8 Modelos)
│   ├── train_8_models.py         # Treino comparativo LOOO vs. Random Split
│   └── Colab_Models_Separated.ipynb # Notebook modular para Kaggle / Colab
│
├── 📂 data/                      # 🎥 GABARITOS E VÍDEOS DE SIMULAÇÃO
│   ├── belt_ground_truth.json    # Mapeamento do gabarito em tempo real
│   ├── simulation_belt.mp4       # Vídeo da esteira rolante de testes
│   └── esteira_textura.jpg       # Textura base da esteira
│
├── 📂 legacy_experiments/        # 📦 ARQUIVO DE CÓDIGOS LEGADOS E TESTES BASELINE
│   ├── train_models.py
│   ├── classify_stream.py
│   └── (scripts legados de experimentos de fase 1)
│
├── 📂 outputs/                   # 💾 RESULTADOS E MÉTRICAS SALVAS AUTOMATICAMENTE
│   └── (gerado automaticamente pelos treinamentos e simulação)
│
├── pipeline.py                   # 🎛️ ORQUESTRADOR CENTRAL VIA LINHA DE COMANDO
├── requirements.txt              # Dependências do projeto
└── README.md                     # Manual de instruções do repositório
```

---

## 🛠️ Como Utilizar o Pipeline Automatizado (`pipeline.py`)

O repositório possui um orquestrador unificado via linha de comando (`pipeline.py`).

### 1. Ingestão de Novas Imagens e Data Quality Gate
Padroniza novas imagens brutas no formato oficial do WaDaBa (`<ID_OBJETO>_a<CODIGO_CLASSE>_<HASH>.jpg`) vinculando o ID do objeto para a divisão **LOOO (Leave-One-Object-Out)**:

```bash
# Ingerir imagens em lote:
python pipeline.py ingest --incoming ./novas_imagens --dataset_dir ./Dataset_Wadaba

# Auditoria de integridade do dataset (Data Quality Gate):
python pipeline.py quality
```

### 2. Treinamento Resiliente e Salvamento Automático
Permite treinar modelos de forma **individual** (com checkpoints por época para evitar perda de progresso por desconexão no Kaggle/Colab):

```bash
# Treinar apenas um modelo específico:
python pipeline.py train --model resnet_looo --epochs 50

# Treinar todos os 8 modelos sequencialmente:
python pipeline.py train --epochs 50
```

Cada treino gera automaticamente em `outputs/<MODELO>/`:
- `model.pth`: Pesos otimizados.
- `split_info.json`: Lista exata dos objetos de treino e dos reservados para teste (`val_objects`).
- `confusion_matrix.png` & `history_plot.png`: Gráficos de alta resolução.
- `metrics_report.txt` & `metrics.json`: Relatório de Precision, Recall e F1-Score.

### 3. Leaderboard e Exportação ONNX
```bash
# Compilar tabela classificatória automática dos modelos treinados:
python pipeline.py leaderboard

# Exportar modelo para ONNX (inferência ultrarrápida):
python pipeline.py export --model_path outputs/modelo_resnet_looo/model.pth --arch resnet
```

### 4. Simulação da Esteira em Tempo Real com Gabarito
Executa a inferência gráfica em tempo real sobre a esteira rolante, comparando a predição com o Gabarito (*Ground Truth*) e indicando objetos inéditos no LOOO:

```bash
# Modo Gráfico (OpenCV):
python pipeline.py simulate --model_path outputs/modelo_resnet_looo/model.pth --arch resnet

# Modo Headless (Avaliação Rápida sem janela gráfica):
python pipeline.py simulate --model_path outputs/modelo_resnet_looo/model.pth --arch resnet --headless
```

---

## 🔬 Pesquisa Acadêmica (Benchmark de 8 Modelos)

Para rodar os scripts do estudo comparativo de **Data Leakage (LOOO vs. Random Split)** apresentados no capítulo de Resultados da dissertação:

```bash
python benchmark/train_8_models.py --data_dir /caminho/para/Dataset_Wadaba
```

---

## ⚙️ Instalação das Dependências

```bash
pip install -r requirements.txt
```
