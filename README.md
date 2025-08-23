## Introduction

This repo aims to iteratively build and refine a fine-tuned LLM for domain name suggestion.

A domain name has to satisfy the following criteria:

1. Relevance & Clarity
2. Simplicity & Memorability
3. ASCII-only, avoid numbers hypens
4. Succint: Length <= 15
5. Audience and Culture Fit
6. Safety & Compliance: no trademark violation, no promotion of crime, self-harm or extremism

## My pipeline

1. Formulate the basic model structure and training module
2. Run Evaluation, Observe and make necessary adaptations
3. Refine the training process


## HyperParameters

MAX_NEW_TOKENS = 480

This is maximum number of tokens the model is allowed to add

MIN_NEW_TOKENS = 150

TEMPERATURE    = 0.7

decides the diversity of response

TOP_P          = 0.92

REPETITION_PENALTY = 1.05

## Training

I utilize QLoRA SfT and trained for 2-3 epochs. Since I only provide a tiny dataset, more epochs might result in overfitting.  

Requiring the exact number of suggestions would be more helpful than requiring more than certain amount, due to the nature of prompt engineering. 



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

Download my model at https://huggingface.co/GeorgesMiradaHas/domain-suggester-qwen25-3b/

## Evalaution

Our Evaluation concerns a variety of domain topics.

Edge case cover usage of emojis, extraordinary length requirement

Underperformance Cases:

1. Parse Error
This is the most common type of error that model output is invalid json format.
Besides, im_end symbol still persists. 

2. Overeaction on Security
e.g. Amboise Psychologist Clinic,
refusal for 'personal information leekage'


3. Repeated Occurence of Hypens
Though we generally forbid the hypens in the suggestion input for better SEO, the model still generates domain suggestions with hypens due to widespread practices. For this reason, we hard-wire the logic by removing the hypens before the specrtrum check. 
e.g. Central Public Hospital, receives:
refusal for 'too_long'

4. Capital Letter suggestion
Domain names are case insensitive, however, to conform with format, we lower all letters in the domain name.
5. Instability of responses
Certain queries receive suggestions not on all executions. 
7. 


## Incorrect Format Guardrail

There is a JSON Guard against non-JSON, truncated JSON, wrong keys.

## Harmful Content Refusal

The LLM inherent capacity enables identification of harmful content. It attaches reason of refusal in returning response. 
Categories refused include:
1. Hate/harassment,
2. Extremist content,
3. illegal goods/services,
4. Sexual content (esp. minors),
5. Self-harm,
6. medical/financial deception,
7. phishing/impersonation, doxxing/PII
