# Projeto de Reconhecimento de Polímeros Plásticos (WaDaBa)

Este repositório contém o código completo para treinamento e avaliação de modelos de inteligência artificial aplicados à classificação de 5 tipos de materiais poliméricos plásticos (**PET, PE_HD, PP, PS e Other**), utilizando a divisão estrita por objeto (*Leave-One-Object-Out / Group Split*).

---

## 📋 Pré-requisitos

- **Python**: versão 3.9 ou superior.
- **Hardware**: GPU recomendada (NVIDIA CUDA ou AMD DirectML/Windows, ou CPU).

---

## 🚀 Passo a Passo para Configuração e Treinamento

### 1. Clonar o Repositório
```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd PlasticRecognitionLocal
```

### 2. Criar e Ativar o Ambiente Virtual

- **No Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **No Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```

> **Nota para GPUs AMD (Windows):** Se você possui uma placa de vídeo AMD Radeon e deseja aceleração por hardware no Windows, instale também o `torch-directml`:
> ```bash
> pip install torch-directml
> ```

---

## 📁 Estrutura do Dataset

Como o dataset completo do WaDaBa possui vários Gigabytes, ele **não fica armazenado dentro do GitHub**.

1. Baixe a pasta descompactada `Dataset_Wadaba` ou o arquivo `Dataset_Wadaba.zip`.
2. Coloque a pasta `Dataset_Wadaba` diretamente na raiz deste repositório:

```text
PlasticRecognitionLocal/
├── train_models.py
├── classify_stream.py
├── requirements.txt
├── README.md
└── Dataset_Wadaba/          <-- Insira a pasta do dataset aqui
    ├── 01_PET/
    ├── 02_PE_HD/
    ├── 05_PP/
    ├── 06_PS/
    └── 07_Other/
```

---

## 🏋️‍♂️ Rodando o Treinamento

Para iniciar o treinamento de todas as 4 arquiteturas por 50 épocas:

```bash
python train_models.py
```

O script irá:
1. Detectar automaticamente sua GPU (NVIDIA CUDA, AMD DirectML ou CPU).
2. Carregar e aplicar a divisão *Leave-One-Object-Out*.
3. Treinar os 4 modelos (`CustomCNN`, `ResNet-50`, `ConvNeXt-Tiny`, `Swin Transformer`).
4. Salvar os arquivos de pesos `.pth`, gráficos de convergência e relatórios de classificação na pasta atual.

---

## 🎥 Rodando a Inferência / Simulação na Esteira

Após o término do treinamento, você pode testar o modelo gerado em um vídeo de esteira executando:

```bash
# Para a ResNet-50:
python classify_stream.py --source simulation_belt.mp4 --model_type resnet --weights modelo_transfer_v2.pth

# Para a ConvNeXt-Tiny:
python classify_stream.py --source simulation_belt.mp4 --model_type convnext --weights modelo_convnext_v2.pth
```
