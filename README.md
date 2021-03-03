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

### Feature Extraction Examples:

```python
from transformers_keras import Bert

# Used to predict directly
model = Bert.from_pretrained('/path/to/pretrained/bert/model')
# segment_ids and mask inputs are optional
sequence_outputs, pooled_output = model.predict((input_ids, segment_ids, mask))
# or
sequence_outputs, pooled_output = model(inputs=(input_ids, segment_ids, mask))

```

Also, you can optionally get the hidden states and attention weights of each encoder layer:

```python
from transformers_keras import Bert

# Used to predict directly
model = Bert.from_pretrained(
    '/path/to/pretrained/bert/model', 
    return_states=True, 
    return_attention_weights=True)
# segment_ids and mask inputs are optional
sequence_outputs, pooled_output, states, attn_weights = model.predict((input_ids, segment_ids, mask))
# or
sequence_outputs, pooled_output, states, attn_weights = model(inputs=(input_ids, segment_ids, mask))

```

### Fine-tuning Examples

```python
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

### Feature Extraction Examples

```python
from transformers_keras import Albert

# Used to predict directly
model = Albert.from_pretrained('/path/to/pretrained/albert/model')
# segment_ids and mask inputs are optional
sequence_outputs, pooled_output = model.predict((input_ids, segment_ids, mask))
# or
sequence_outputs, pooled_output = model(inputs=(input_ids, segment_ids, mask))
```

Also, you can optionally get the hidden states and attention weights of each encoder layer:

```python
from transformers_keras import Albert

# Used to predict directly
model = Albert.from_pretrained(
    '/path/to/pretrained/albert/model', 
    return_states=True, 
    return_attention_weights=True)
# segment_ids and mask inputs are optional
sequence_outputs, pooled_output, states, attn_weights = model.predict((input_ids, segment_ids, mask))
# or
sequence_outputs, pooled_output, states, attn_weights = model(inputs=(input_ids, segment_ids, mask))
```

### Fine-tuing Examples

```python

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

## Advanced Usage

Here are some advanced usages:

* Skip loadding weights from checkpoint
* Load other pretrained models

### Skip loadding weights from checkpoint

You can skip loadding some weights from ckpt.

Examples:

```python
from transformers_keras import Bert, Albert

ALBERT_MODEL_PATH = '/path/to/albert/model'
albert = Albert.from_pretrained(
    ALBERT_MODEL_PATH,
    # return_states=False,
    # return_attention_weights=False,
    skip_token_embedding=True,
    skip_position_embedding=True,
    skip_segment_embedding=True,
    skip_pooler=True,
    ...
    )

BERT_MODEL_PATH = '/path/to/bert/model'
bert = Bert.from_pretrained(
    BERT_MODEL_PATH,
    # return_states=False,
    # return_attention_weights=False,
    skip_token_embedding=True,
    skip_position_embedding=True,
    skip_segment_embedding=True,
    skip_pooler=True,
    ...
    )
```

All supported kwargs to skip loadding weights:

* `skip_token_embedding`, skip loadding `token_embedding` weights from ckpt
* `skip_position_embedding`, skip loadding `position_embedding` weights from ckpt
* `skip_segment_embedding`, skip loadding `token_type_emebdding` weights from ckpt
* `skip_embedding_layernorm`, skip loadding `layer_norm` weights of emebedding layer from ckpt
* `skip_pooler`, skip loadding `pooler` weights of pooler layer from ckpt



### Load other pretrained models

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
