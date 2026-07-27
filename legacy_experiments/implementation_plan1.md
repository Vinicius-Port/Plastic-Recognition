# Plano de Melhoria dos Modelos de Classificação de Plástico

Analisei o arquivo PDF com os resultados do treinamento. Aqui está o que eu descobri e como podemos melhorar o algoritmo `train_models.py`.

## O que os resultados nos mostraram?

1. **A Classe `PET` domina o dataset (Desbalanceamento):**
   - Na sua base de validação, a classe `PET` tem 423 amostras, enquanto a classe `Other` tem apenas 11.
   - Isso fez com que a **CNN Customizada** tivesse 0% de acerto na classe `Other` (pois ela focou em acertar o que aparecia mais).
   - Pior ainda, o modelo de **Transfer Learning (MobileNetV2)** "desistiu" de aprender e começou a chutar que **quase todas as imagens eram `PET`**. Isso garantiu a ele ~55% de acurácia, mas a precisão para `PE_HD`, `PP` e `PS` despencou para quase zero.

2. **O Transfer Learning (MobileNetV2) parou muito cedo:**
   - Ele deu *Early Stopping* na época 5.
   - O *Learning Rate* estava muito baixo (`0.0001`) para treinar a última camada do zero. Além disso, os pesos principais do MobileNetV2 estavam totalmente congelados, impedindo que o modelo se adaptasse aos tipos de plásticos específicos.

## Proposta de Melhoria (O que vamos alterar no `train_models.py`)

Para resolver esses problemas, proponho aplicarmos as seguintes melhorias:

### 1. Aplicar Pesos de Classe (`class_weight`)
Vamos calcular dinamicamente o peso de cada classe no conjunto de treinamento. Classes com poucas imagens (como `Other`) receberão um "peso maior" na hora de calcular o erro (loss), forçando a rede neural a prestar atenção nelas e não apenas na classe majoritária (`PET`). Passaremos isso para o `model.fit()`.

### 2. Melhorar o Transfer Learning (Fine-Tuning em Duas Etapas)
Atualmente o MobileNetV2 está totalmente congelado. Vamos mudar a abordagem:
- **Fase 1:** Treinar apenas a nova camada adicionada no topo com um Learning Rate ligeiramente maior (`1e-3`) para que ela aprenda o básico.
- **Fase 2 (Fine-Tuning real):** Descongelar as últimas 20 a 30 camadas do MobileNetV2 e treiná-las junto com a camada final, usando um Learning Rate bem pequeno (`1e-5`). Isso faz com que a rede se especialize de verdade no seu conjunto de plásticos.

### 3. Adicionar `ReduceLROnPlateau`
Vamos adicionar um *Callback* chamado `ReduceLROnPlateau`. Ele funciona junto com o *EarlyStopping*, mas se o modelo empacar (a *loss* não diminuir), ele reduz a taxa de aprendizado, dando um "ajuste fino" extra antes de o modelo desistir e parar.

> [!IMPORTANT]
> **Revisão Necessária:**
> Essas mudanças farão com que o tempo de treinamento aumente um pouco (principalmente a Fase 2 do Transfer Learning). Se você estiver usando o Google Colab sem GPU ou um PC sem placa de vídeo, pode demorar mais.

Você aprova as mudanças sugeridas para começarmos a aplicá-las no código?
