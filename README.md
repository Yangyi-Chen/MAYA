# RLTextualAttack

This is the official code base for the EMNLP 2021 paper, "[**Multi-granularity Textual Adversarial Attack with Behavior Cloning**](https://arxiv.org/pdf/1912.10375.pdf)".

Here are some brief introductions for main folders.

## models

This folder  saves a file about pre-defined victim model structure (uniform interface to be called) and a sub-folder to save pre-trained victim-models. More details can be found when click in.

## MG

This folder contains  python files to attack victim model and a sub-folder to train a reinforcement learning model. More details can be found when click in.

## Dataset

this folder contains different datasets and provide dataset templates in attack process.

## ADsamples_evaluation

This folder contains python files to evaluate adversarial samples. More details can be found when click in.

## paraphrase_models

This folder saves the GPT2 paraphrase model (not necessary when not use GPT2 model).

## How to use our attack

### Download all folders

### Prepare victim model

When do MAYA attack, you need to implement the uniform interface defined in “models.py”. All you have to do is just to implement the “\__call__” function and there are some victim model classes you can directly use (BiLSTM, BERT and RoBERTa), so you only need to prepare your own victim model (e.g. [SST-2 BERT](https://drive.google.com/drive/folders/1T9dq05YcVluuQ9UpEpg_KHya51HycuYU)).

### Prepare dataset

We use tsv format to save dataset. You can found more information in dataset folder.



### Run attacking code

Now you could start MAYA attack by directly run the code “attack.py”.

```
python3.7 attack.py
```



### Prepare RL model (optional)

We provide several checkpoints of out pre-trained RL model for you to use, 

