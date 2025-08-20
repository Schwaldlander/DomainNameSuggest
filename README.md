## Introduction

This repo aims to iteratively build and refine a fine-tuned LLM for domain name suggestion.

A domain name has to satisfy the following criteria:

1. Relevance & Clarity
2. Simplicity & Memorability
3. ASCII-only, avoid numbers hypens
4. Succint: Length <= 15
5. Audience and Culture Fit
6. Safety & Compliance 


## HyperParameters

MAX_NEW_TOKENS = 420

TEMPERATURE    = 0.7
TOP_P          = 0.92

## Intermediate Training Results

Given LoRA rank $r$, learning rate $lr$ and number of epochs $epochs$

When r = 8,  lr = 2e-4, epochs = 2,
|Epoch	   |Training Loss|	Validation Loss|
|----------|----------|----------|
|1         |	No log  | 	1.395825|
|2         |	No log	|   1.278103|

When r = 16,  lr = 1e-4, epochs = 2,
|Epoch	   |Training Loss|	Validation Loss|
|----------|----------|----------|
|1         |	No log  | 	1.534792|
|2         |	No log	|   1.431465|

When r = 32,  lr = 8e-5, epochs = 3,
|Epoch	   |Training Loss|	Validation Loss|
|----------|----------|----------|
|1         |	No log  | 	1.584584|
|2         |	No log	|   1.460789|
|3         |	No log	|   1.413446|
