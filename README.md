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

# Used to predict directly
model = Bert.from_pretrained('/path/to/pretrained/bert/model')
# segment_ids and mask inputs are optional
model.predict((input_ids, segment_ids, mask))
# or
model(inputs=(input_ids, segment_ids, mask))

# Used to fine-tuning
def build_bert_classify_model(pretrained_model_dir, trainable=True, **kwargs):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    # segment_ids and mask inputs are optional
    segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')

    bert = Bert.from_pretrained(pretrained_model_dir, **kwargs)
    bert.trainable = trainable

    sequence_outputs, pooled_output = bert(inputs=(input_ids, segment_ids))
    outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
    model = tf.keras.Model(inputs=[input_ids, segment_ids], outputs=outputs)
    model.compile(loss='binary_cross_entropy', optimizer='adam')
    return model

model = build_bert_classify_model(
            pretrained_model_dir=os.path.join(BASE_DIR, 'chinese_wwm_ext_L-12_H-768_A-12'),
            trainable=True)
model.summary()
```


## ALBERT

Supported pretrained models:

* All the ALBERT models pretrained by [google-research/albert](https://github.com/google-research/albert)

```python
from transformers_keras import Albert

# Used to predict directly
model = Bert.from_pretrained('/path/to/pretrained/albert/model')
# segment_ids and mask inputs are optional
model.predict((input_ids, segment_ids, mask))
# or
model(inputs=(input_ids, segment_ids, mask))

# Used to fine-tuning 
def build_albert_classify_model(pretrained_model_dir, trainable=True, **kwargs):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    # segment_ids and mask inputs are optional
    segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')

    albert = Albert.from_pretrained(pretrained_model_dir, **kwargs)
    albert.trainable = trainable

    sequence_outputs, pooled_output = albert(inputs=(input_ids, segment_ids))
    outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
    model = tf.keras.Model(inputs=[input_ids, segment_ids], outputs=outputs)
    model.compile(loss='binary_cross_entropy', optimizer='adam')
    return model

model = build_albert_classify_model(
            pretrained_model_dir=os.path.join(BASE_DIR, 'albert_base'),
            trainable=True)
model.summary()
```

## Load other pretrained models

If you want to load models pretrained by other implementationds, whose config and trainable weights are a little different from previous, you can subclass `AbstractAdapter` to adapte these models:

```python
from transformers_keras.adapters import AbstractAdapter
from transformers_keras import Bert, Albert

# load custom bert models
class MyBertAdapter(AbstractAdapter):

    def adapte_config(self, config_file, **kwargs):
        # adapte model config here
        # you can refer to `transformers_keras.adapters.bert_adapter`
        pass

    def adapte_weights(self, model, config, ckpt, **kwargs):
        # adapte model weights here
        # you can refer to `transformers_keras.adapters.bert_adapter`
        pass

bert = Bert.from_pretrained('/path/to/your/bert/model', adapter=MyBertAdapter())

# or, load custom albert models
class MyAlbertAdapter(AbstractAdapter):

    def adapte_config(self, config_file, **kwargs):
        # adapte model config here
        # you can refer to `transformers_keras.adapters.albert_adapter`
        pass

    def adapte_weights(self, model, config, ckpt, **kwargs):
        # adapte model weights here
        # you can refer to `transformers_keras.adapters.albert_adapter`
        pass

albert = Albert.from_pretrained('/path/to/your/albert/model', adapter=MyAlbertAdapter())
```
