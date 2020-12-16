# transformers-keras

![Python package](https://github.com/luozhouyang/transformers-keras/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/transformers-keras.svg)](https://badge.fury.io/py/transformers-keras)
[![Python](https://img.shields.io/pypi/pyversions/transformers-keras.svg?style=plastic)](https://badge.fury.io/py/transformers-keras)

Transformer-based models implemented in tensorflow 2.x(Keras).

## Installation

```bash
pip install -U transformers-keras
```

## Models

- [x] Transformer[*DELETED*]
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 
  * Here is a tutorial from tensorflow:[Transformer model for language understanding](https://www.tensorflow.org/beta/tutorials/text/transformer)
- [x] BERT
  * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [x] ALBERT
  * [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)


## BERT

Supported pretrained models:

* All the BERT models pretrained by [google-research/bert](https://github.com/google-research/bert)
* All the BERT & RoBERTa models pretrained by [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)

```python
from transformers_keras import Bert

model = Bert.from_pretrained('/path/to/pretrained/bert/model')
# then, use this model to fine-tune new model as a keras layer, or to do inference
model(model.dummy_inputs())

```


## ALBERT

Supported pretrained models:

* All the ALBERT models pretrained by [google-research/albert](https://github.com/google-research/albert)

```python
from transformers_keras import Albert

model = Bert.from_pretrained('/path/to/pretrained/albert/model')
# then, use this model to fine-tune new model as a keras layer, or to do inference
model(model.dummy_inputs())
```

## Load other pretrained models

If you want to load pretraine models using other implementationds, whose config and trainable weights are a little different with previous, you can subclass `AbstractAdapter` to adapte these models:

```python
from transformers_keras.adapters import AbstractAdapter
from transformers_keras import Bert, Albert

# load custom bert models
class MyBertAdapter(AbstractAdapter):

  def adapte(self, pretrained_model_dir, **kwargs):
      # you can refer to `transformers_keras.adapters.bert_adapter`
      pass

bert = Bert.from_pretrained('/path/to/your/bert/model', adapter=MyBertAdapter())

# or, load custom albert models
class MyAlbertAdapter(AbstractAdapter):

  def adapte(self, pretrained_model_dir, **kwargs):
      # you can refer to `transformers_keras.adapters.albert_adapter`
      pass
albert = Albert.from_pretrained('/path/to/your/albert/model', adapter=MyAlbertAdapter())
```
