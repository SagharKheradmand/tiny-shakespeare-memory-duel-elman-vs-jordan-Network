# Elman vs Jordan Networks for Tiny Shakespeare

## Overview

This project compares two classical recurrent neural network architectures, the Elman Network and the Jordan Network, on a character-level language modeling task using the Tiny Shakespeare dataset.

The objective is to study how different recurrent feedback mechanisms affect sequence learning and text generation.

The Elman Network feeds the previous hidden state back into the network, while the Jordan Network uses the previous output probability distribution as recurrent context.

Both models are trained under the same experimental settings and evaluated using training and validation loss, character-level accuracy, perplexity, convergence behavior, and generated text quality.

The effect of sampling temperature is also investigated by generating text at different temperature values.

---

# Problem

Given a sequence of characters:

```text
ROMEO:
```

the model learns to predict the next character at every time step.

Instead of predicting complete words, the text is treated as a character-level sequence.

For example:

```text
Input:   R O M E O :
Target:  O M E O : \n
```

By repeatedly predicting the next character, the trained model can generate new text with patterns similar to Shakespearean writing.

---

# Dataset

The project uses the Tiny Shakespeare dataset.

The dataset contains Shakespearean text and is treated as a character-level corpus.

To reduce computational requirements, the implementation supports training on only a fraction of the complete dataset.

The default configuration uses:

```text
30% of the Tiny Shakespeare dataset
```

The selected text is divided into:

```text
90% Training
10% Validation
```

A vocabulary is automatically constructed from all unique characters appearing in the selected portion of the dataset.

Each character is mapped to an integer index before being passed to the neural networks.

---

# Models

Two recurrent architectures are implemented and compared.

## Elman Network

The Elman Network uses the previous hidden state as recurrent memory.

Its hidden state is updated according to:

```text
h_t = tanh(W_xh x_t + W_hh h_(t-1) + b)
```

The output is then calculated from the current hidden state:

```text
y_t = W_hy h_t
```

The recurrent connection can therefore be represented as:

```text
Input x_t
    |
    v
Hidden h_t ----> Output y_t
    ^
    |
Hidden h_(t-1)
```

The previous hidden representation acts as the network's memory of earlier characters.

---

## Jordan Network

The Jordan Network uses the previous output distribution instead of the previous hidden state.

Its hidden state is calculated as:

```text
h_t = tanh(W_xh x_t + W_yh y_(t-1) + b)
```

and the output is:

```text
y_t = W_hy h_t
```

The recurrent structure is therefore:

```text
Input x_t
    |
    v
Hidden h_t ----> Output y_t
    ^                |
    |________________|
      previous output
```

In this implementation, the previous output probability distribution remains part of the differentiable computation during training.

This allows gradients to propagate through the Jordan feedback path.

---

# Elman vs Jordan

The main architectural difference is the source of recurrent information.

| Architecture | Recurrent Feedback |
| --- | --- |
| Elman Network | Previous hidden state |
| Jordan Network | Previous output distribution |

The Elman architecture directly preserves an internal hidden representation of previous inputs.

The Jordan architecture instead uses information contained in the previous prediction.

The experiment investigates how this difference affects:

- Training convergence
- Validation performance
- Character prediction accuracy
- Perplexity
- Stability
- Generated text quality

---

# Experimental Configuration

The default configuration used by the implementation is:

| Parameter | Value |
| --- | --- |
| Dataset fraction | 30% |
| Train ratio | 90% |
| Embedding dimension | 64 |
| Hidden size | 128 |
| Sequence length | 100 |
| Batch size | 64 |
| Epochs | 15 |
| Steps per epoch | 100 |
| Evaluation iterations | 30 |
| Learning rate | 0.001 |
| Generated characters | 500 |
| Gradient clipping | 1.0 |
| Random seed | 42 |

The implementation automatically uses CUDA when a compatible GPU is available and otherwise runs on CPU.

---

# Training

Both models are trained under the same experimental conditions to provide a fair comparison.

The training pipeline consists of:

1. Loading the Tiny Shakespeare text
2. Selecting the configured fraction of the dataset
3. Building the character vocabulary
4. Encoding characters as integer indices
5. Splitting the sequence into training and validation sets
6. Creating random character sequences
7. Training the Elman Network
8. Training the Jordan Network
9. Evaluating both models after every epoch
10. Saving the best model according to validation loss
11. Generating text using the trained models
12. Comparing different sampling temperatures

---

# Optimization

Both networks are trained using the Adam optimizer.

```text
Learning rate = 0.001
```

Character prediction is treated as a multi-class classification problem over the character vocabulary.

Cross-entropy loss is used as the training objective.

Gradient clipping is also applied:

```text
Maximum gradient norm = 1.0
```

This helps reduce the risk of exploding gradients during recurrent training.

---

# Evaluation Metrics

The two architectures are evaluated using several metrics.

## Cross-Entropy Loss

Cross-entropy measures how well the predicted probability distribution matches the correct next character.

Lower values indicate better predictions.

Both training and validation loss are recorded after every epoch.

---

## Character Accuracy

Character-level accuracy measures the proportion of positions where the character with the highest predicted probability matches the actual next character.

```text
Accuracy =
Correct Character Predictions
-----------------------------
Total Character Predictions
```

Higher character accuracy indicates better next-character prediction.

---

## Perplexity

Perplexity is calculated from cross-entropy loss:

```text
Perplexity = exp(Loss)
```

A lower perplexity means the model is less uncertain about the next character.

For language modeling, perplexity provides a useful interpretation of predictive uncertainty.

---

# Training Curves

The implementation automatically saves several plots for comparing the two architectures.

## Training Loss

```text
result/train_loss(1).png
```

This plot compares how quickly the Elman and Jordan networks reduce their training loss.

---

## Validation Loss

```text
result/val_loss(1).png
```

Validation loss is used to evaluate how well each architecture generalizes to unseen character sequences.

---

## Training Accuracy

```text
result/train_acc.png
```

This plot shows the character-level training accuracy across epochs.

---

## Validation Accuracy

```text
result/val_acc(1).png
```

Validation accuracy provides a direct comparison of next-character prediction performance.

---

## Training Perplexity

```text
result/train_ppl.png
```

This plot shows how uncertainty on the training sequences changes during optimization.

---

## Validation Perplexity

```text
result/val_ppl(1).png
```

Validation perplexity is particularly useful for comparing the generalization ability of the two recurrent architectures.

---

# Text Generation

After training, both models are used to generate new character sequences.

The same seed text is used for both architectures:

```text
ROMEO:
```

Using the same seed provides a fair qualitative comparison between the Elman and Jordan networks.

For each generated character, the model produces a probability distribution over the vocabulary.

A character is then sampled from this distribution and used as input for the next generation step.

---

# Temperature Sampling

Temperature controls the randomness of text generation.

The project evaluates three temperatures:

```text
0.4
0.8
1.2
```

## Temperature = 0.4

A low temperature sharpens the probability distribution.

The model strongly prefers characters with high predicted probability.

Typical behavior:

- More conservative output
- More predictable character sequences
- Better local consistency
- Greater repetition
- Lower diversity

---

## Temperature = 0.8

A moderate temperature provides a compromise between stability and randomness.

Typical behavior:

- More natural variation
- Reasonable structural consistency
- Less repetition
- Better balance between coherence and diversity

---

## Temperature = 1.2

A high temperature produces a flatter probability distribution.

Lower-probability characters therefore have a greater chance of being selected.

Typical behavior:

- Greater diversity
- More unexpected sequences
- More creative output
- Higher probability of spelling or structural errors
- Lower overall coherence

---

# Temperature Comparison

| Temperature | Randomness | Stability | Diversity | Expected Behavior |
| ---: | --- | --- | --- | --- |
| 0.4 | Low | High | Low | Conservative and repetitive |
| 0.8 | Medium | Medium/High | Medium | Balanced generation |
| 1.2 | High | Low | High | Diverse but noisier |

This experiment demonstrates the trade-off between deterministic and creative generation.

---

# Comparison Criteria

The Elman and Jordan architectures are compared from several perspectives.

## Convergence Speed

Training and validation loss curves are used to determine which model learns faster.

A model that reaches a low validation loss in fewer epochs can be considered to have faster convergence.

---

## Stability

The smoothness of loss, accuracy, and perplexity curves provides information about optimization stability.

Large fluctuations can indicate more unstable learning behavior.

---

## Validation Performance

Validation loss, character accuracy, and perplexity are used to evaluate generalization.

The architecture with:

- Lower validation loss
- Higher validation accuracy
- Lower validation perplexity

provides stronger quantitative performance.

---

## Generated Text Quality

Numerical metrics alone cannot completely describe a language model.

Generated text is therefore inspected qualitatively.

Important characteristics include:

- Word-like character patterns
- Spacing
- Punctuation
- Line structure
- Repetition
- Local coherence
- Diversity

The same seed and temperatures are used for both architectures so their generated outputs can be compared directly.

---

# Output Files

The implementation can automatically save:

```text
training_history.csv
config.json
report_summary.md
best_elman_model.pt
best_jordan_model.pt
final_elman_model.pt
final_jordan_model.pt
```

Generated text samples are saved separately for every model and temperature.

Example:

```text
generated_samples/
├── elman_temperature_0.4.txt
├── elman_temperature_0.8.txt
├── elman_temperature_1.2.txt
├── jordan_temperature_0.4.txt
├── jordan_temperature_0.8.txt
└── jordan_temperature_1.2.txt
```

---

# Project Structure

```text
elman-vs-jordan-tiny-shakespeare/
│
├── README.md
├── p2_completed.py
│
├── result/
│   ├── train_acc.png
│   ├── train_loss(1).png
│   ├── train_ppl.png
│   ├── val_acc(1).png
│   ├── val_loss(1).png
│   └── val_ppl(1).png
│
└── report.pdf
```

---

# File Description

| File | Description |
| --- | --- |
| `p2_completed.py` | Complete implementation of the Elman and Jordan recurrent networks |
| `train_loss(1).png` | Training loss comparison |
| `val_loss(1).png` | Validation loss comparison |
| `train_acc.png` | Training character accuracy |
| `val_acc(1).png` | Validation character accuracy |
| `train_ppl.png` | Training perplexity comparison |
| `val_ppl(1).png` | Validation perplexity comparison |
| `report.pdf` | Detailed assignment report |

---

# Running the Project

The project requires Python and PyTorch.

Install the main dependencies:

```bash
pip install torch matplotlib
```

Place the Tiny Shakespeare text file beside the Python script.

For example:

```text
tiny-shakespeare.txt
```

Then run:

```bash
python p2_completed.py --data_path tiny-shakespeare.txt
```

A custom dataset path can also be provided:

```bash
python p2_completed.py --data_path "path/to/tiny-shakespeare.txt"
```

---

# Custom Experiments

The implementation provides command-line arguments for changing the experimental settings.

For example:

```bash
python p2_completed.py \
    --data_path tiny-shakespeare.txt \
    --epochs 20 \
    --hidden_size 128 \
    --batch_size 64
```

The fraction of the dataset can also be changed:

```bash
python p2_completed.py \
    --data_path tiny-shakespeare.txt \
    --data_fraction 0.50
```

Different sampling temperatures can be tested using:

```bash
python p2_completed.py \
    --data_path tiny-shakespeare.txt \
    --temperatures 0.4 0.8 1.2
```

---

# Technologies and Methods

- Python
- PyTorch
- Matplotlib
- Recurrent Neural Networks
- Elman Network
- Jordan Network
- Character-Level Language Modeling
- Text Generation
- Embedding Layers
- Cross-Entropy Loss
- Adam Optimizer
- Gradient Clipping
- Perplexity
- Temperature Sampling

---

# Key Concepts Demonstrated

This project demonstrates several important concepts in neural networks and sequence modeling:

- Recurrent neural networks
- Temporal memory
- Hidden-state feedback
- Output-distribution feedback
- Character-level language modeling
- Next-character prediction
- Sequence learning
- Recurrent optimization
- Gradient clipping
- Language-model perplexity
- Autoregressive text generation
- Temperature-based sampling
- Quantitative model comparison
- Qualitative evaluation of generated text

---

# Conclusion

This project provides a direct comparison between two classical recurrent neural network architectures for character-level language modeling.

Although both Elman and Jordan networks are recurrent models, they preserve temporal information in different ways. The Elman Network feeds its previous hidden representation back into the next time step, whereas the Jordan Network uses the previous output probability distribution.

Training both models under identical conditions makes it possible to investigate how these different memory mechanisms affect convergence, predictive accuracy, perplexity, and text generation.

The temperature experiments also demonstrate that text-generation quality is controlled not only by the trained model but also by the sampling strategy. Lower temperatures favor stability and repetition, while higher temperatures increase diversity at the cost of coherence.

Overall, the project illustrates how recurrent feedback design influences sequential learning and provides a practical comparison of two foundational recurrent neural network architectures.

---

# Course Information

**Course:** Neural Networks and Deep Learning  
**Assignment:** 2 - Tiny Shakespeare Memory Duel  
**University:** Shiraz University  
**Year:** 2026

---

# Author

Saghar Kheradmand
