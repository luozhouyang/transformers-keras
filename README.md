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

All the bert models pretrained by [google-research/bert](https://github.com/google-research/bert) can be loaded: 

```python
from transformers_keras import Bert

model = Bert.from_pretrained('/path/to/pretrained/bert/model')
# then, use this model to fine-tune new model as a keras layer, or to do inference
model(model.dummy_inputs())

```


## ALBERT

All the albert models pretrained by [google-research/albert](https://github.com/google-research/albert) can be loaded: 

```python
from transformers_keras import Albert

model = Bert.from_pretrained('/path/to/pretrained/albert/model')
# then, use this model to fine-tune new model as a keras layer, or to do inference
model(model.dummy_inputs())
```


