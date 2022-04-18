# MAYA

This is the official code base for the EMNLP 2021 paper, "[**Multi-granularity Textual Adversarial Attack with Behavior Cloning**](https://aclanthology.org/2021.emnlp-main.371.pdf)".

Here are some brief introductions for main folders.

## models

This folder  saves a file about pre-defined victim model structure (uniform interface to be called) and a sub-folder to save pre-trained victim-models.

## MG

This folder contains  python files to attack victim model and a sub-folder to train a reinforcement learning model. 

## Dataset

this folder contains different datasets and provide dataset templates in attack process.

## ADsamples_evaluation

This folder contains python files to evaluate adversarial samples.

## paraphrase_models

This folder saves the GPT2 paraphrase model (not necessary when not use GPT2 model).

## How to use our attack

### Installation

```
git clone https://github.com/Yangyi-Chen/MAYA
```

### Download GPT2 model(optional)

https://drive.google.com/drive/folders/1RmiXX8u1ojj4jorj_QgxOWWkryDIdie-

If you want to use the GPT2 paraphraser, download these files and put all downloaded files in the“paraphrase_models/style_transfer_paraphrase/”directory.

### Prepare victim model

When do MAYA attack, you need to implement the uniform interface defined in “models.py”. All you have to do is just to implement the “\__call__” function and there are some victim model classes you can directly use (BiLSTM, BERT and RoBERTa), so you only need to prepare your own victim model .

You can downloaded pretrained victim model here([SST-2 BERT](https://drive.google.com/drive/folders/1T9dq05YcVluuQ9UpEpg_KHya51HycuYU)), put it in “models/pretrained_models/bert_for_sst2”.

### Prepare dataset

We use tsv format to save dataset. You can found more information in “dataset” folder.

### Register an account for Baidu Translation API

You could visit https://fanyi-api.baidu.com/product/11 to get registration and then apply for your application id and secret key. And then fill them in “BaiduTransAPI_forPython3.py”in “MG” folder.

### Run attacking code

Now you could start MAYA attack by directly run the code “attack.py”.

You could set hyperparameters in “attack.py”such as ways of paraphrase, number of attacks, etc. 

```
python attack.py
```

#### **Note**: Some packages are needed when running the python file. Just install these packages following corresponding prompts.

### Prepare RL model (optional)

We provide several checkpoints of our pre-trained RL model for you to use (https://drive.google.com/drive/folders/1GfWs8YN9hRPwN7CmTrgfuh028KWqrH2M).

You could get how to use it in “attack.py”.



## Citation

If you find it useful, please cite the following work:



```
@article{chen2021multi,
  title={Multi-granularity Textual Adversarial Attack with Behavior Cloning},
  author={Chen, Yangyi and Su, Jin and Wei, Wei},
  journal={arXiv preprint arXiv:2109.04367},
  year={2021}
}
```

