
# Technical Report 


## Introduction

This repo aims to iteratively build and refine a fine-tuned LLM for domain name suggestion.

A domain name has to satisfy the following criteria:

1. Relevance & Clarity
2. Simplicity & Memorability
3. ASCII-only, avoid numbers hypens
4. Succint: Length <= 15
5. Audience and Culture Fit
6. Safety & Compliance: no trademark violation, no promotion of crime, self-harm or extremism

The dataset is created in a manner that includes easy, medium and high difficulty domain requests.
Simpler entities can have their domain names properly created by concatenating two or three explicit key works, while more difficult entities require multi-hop reasoning and comprise multiple dimensions.

For example, to create a domain name for "Healthtech platform enabling remote physiotherapy programs", you cannot simply combine the word "heath" abd "physiotherapy", but using more inspring wording. 
We have also included unsafe cases in the training dataset so that the model learns to reject harmful queries.

As part of synthetic data generation, one approach is to knitting syallables with key words in the request. 
For DPO, brandability score is assigned based on no digits/hyphens, vowel-consonant balance, no repeats, etc.
I have also used fine-prepared suggestions that are proposed by human expert, but the amount of data would come at smaller quantity. 




## Model Development and Iteration

Our repo centers on Open Source Models like **meta-llama/Llama-3.2-3B-Instruct** as a baseline.
Other models such as **Qwen2.5-3B-Instruct**, **mistralai/Mistral-7B-Instruct-v0.3** are also workable. Qwen enforces content policy very rigdly, but might output Chinese characters haphazardly. On the other hand, **meta-llama/Llama-3.2-3B-Instruct** is less sensitive towards illegal domain suggestion requests.

For efficient usage of memory and RAM, we use quantized LoRA. Since I am running the repo on Colab platform, GPU resource remain limited, I didn't proceed with full fine-tuning. 
I utilize QLoRA SfT and trained for 2-3 epochs. Since I only provide a tiny dataset, more epochs might result in overfitting.  

It is important to formulate prompts in a clear and concise manner. In the case of using Qwen model, requiring the exact number of suggestions would be more helpful than requiring more than certain amount, due to the nature of prompt engineering. 

The **pipeline** is as follows:
1. Formulate the basic model structure and training module
2. Run Evaluation, Store model checkpoints in folders with timestamps
3. Observe and make necessary adaptations
4. Refine the training process

Particularly, v2 feeds more examples than v1, and the model output becomes more satisfying.
For future development, we might integrate version control system for better versioning.

### HyperParameters

MAX_NEW_TOKENS

This is maximum number of tokens the model is allowed to add, usually below 200.

TEMPERATURE    = 0.45

decides the diversity of response

TOP_P          = 0.92

selects the items with possibilities that equal or exceed this value

REPETITION_PENALTY = 1.05

avoid autoregressive failure



### Intermediate Training Results

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

### SfT vs DPO

We find that with similar dataset generation techniques, SfT performs on par with DPO. Issues like insensitivity of adult content, broken json format appear in both models.

## Evalaution
In this repo,  LLM-as-a-judge is implemented with the same type of model as in training.
Our evaluation covers similar topics as in training dataset.
Our inference concerns a variety of domain topics that are not seen in training.

Edge case cover usage of emojis, extraordinary length requirement.

### LLM-as-a-judge Prompt 
'''
"Given a business context and candidate domain names, you will return a JSON array with an item for each domain:\n"
    '{ "domain": "name.tld", "score": 0..1, "generic": true|false, "reason": "short human-readable reason" }.\n'
    "Scoring rubric (0..1):\n"
    "- 0.9–1.0: Distinctive, highly brandable, short, memorable.\n"
    "- 0.7–0.89: Strong; fits business; not generic; pronounceable.\n"
    "- 0.5–0.69: Acceptable but somewhat bland or derivative.\n"
    "- <0.5: Unappealing, generic, spammy, too long, or violates constraints.\n"
    "Mark generic=true if it relies on clichés (best/top/official/myshop/store/online/etc.) or is overly literal.\n"
    "Keep answers concise."
'''

### Underperformance Cases

- 1. Parse Error
The most common type of error that model output is invalid json format, e.g. brackets{([ fail to match or close.
Besides, irregular symbol such as im_end still persists.

**Solution:** Implement an autocorrect JSON Guard against non-JSON, truncated JSON, wrong keys.
 

 - 2. Inappropriate reaction on Security


    1) Qwen diisplays over security.
    e.g. Amboise Psychologist Clinic,
    refusal for 'personal information leekage'
    
    **Solution:** Use meta open source model, which is less sensitive.
    
    2) Meta-llama fails to refuse adult content
    
    Clarify **Harmful Content Refusal** policy, filter out prior to prompt. [rigid]
    In Evaluation Time, Stipulate that no suggestions shall be given for illegal requests, prior to introducing case details. [adaptive]
    
    The LLM inherent capacity enables identification of harmful content. It attaches reason of refusal in returning response. 
    Categories refused include:
    1. Hate/harassment,
    2. Extremist content,
    3. illegal goods/services,
    4. Sexual content (esp. minors),
    5. Self-harm,
    6. medical/financial deception,
    7. phishing/impersonation, doxxing/PII

 - 3. Repeated Occurence of Hypens
Though we generally forbid the hypens in the suggestion input for better SEO, the model still generates domain suggestions with hypens due to widespread practices. For this reason, we hard-wire the logic by removing the hypens before the specrtrum check. 
e.g. Central Public Hospital, receives:
refusal for 'too_long'

**Solution:** Meta model performs better in avoiding hypens as well. In case of hypen/space appearance, we can force remove such tokens.

 - 4. Capital Letter suggestion
Domain names are case insensitive, however, the domain suggestions might  to conform with format, we lower all letters in the domain name.

**Solution:** not vital issue

 - 5. Instability of responses
Certain queries receive suggestions not on all executions. This is evident when model.eval() is not activated.

**Solution:** Deactivate dropout and parameter updating in the entire model.

 - 6. Generic Suggestion

Some domain suggestions include health.io, online.ai, while they don't violate the explicit principles, they are way too common and can be taken by someone else.

**Solution:** Add uniqueness requirement in evaluation prompts



