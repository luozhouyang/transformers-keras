# transformers-keras

![Python package](https://github.com/luozhouyang/transformers-keras/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/transformers-keras.svg)](https://badge.fury.io/py/transformers-keras)
[![Python](https://img.shields.io/pypi/pyversions/transformers-keras.svg?style=plastic)](https://badge.fury.io/py/transformers-keras)


基于`tensorflow 2.x(Keras)` 实现的多种`Transformer`模型，但是不仅仅是`Transformer`! 可以无缝**加载预训练模型**进行微调下游任务！

[English](README.md) | [中文文档]

- [transformers-keras](#transformers-keras)
  - [安装](#安装)
  - [实现的模型](#实现的模型)
  - [Transformer的使用](#transformer的使用)
  - [BERT的使用](#bert的使用)
    - [从头预训练BERT](#从头预训练bert)
    - [加载预训练好的BERT模型](#加载预训练好的bert模型)
      - [加载完整的`BERT`模型](#加载完整的bert模型)
      - [加载`BERT`的`Encoder`部分](#加载bert的encoder部分)
  - [ALBERT](#albert)

## 安装

```bash
pip install -U transformers-keras
```

## 实现的模型

- [x] Transformer
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 
  * Here is a tutorial from tensorflow:[Transformer model for language understanding](https://www.tensorflow.org/beta/tutorials/text/transformer)
- [x] BERT
  * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [x] ALBERT
  * [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)


## Transformer的使用

训练一个`transformer`:

```python
from transformers_keras import TransformerTextFileDatasetBuilder
from transformers_keras import TransformerDefaultTokenizer
from transformers_keras import TransformerRunner


src_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_src.txt')
tgt_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_tgt.txt')
dataset_builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer)

model_config = {
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'src_vocab_size': src_tokenizer.vocab_size,
    'tgt_vocab_size': tgt_tokenizer.vocab_size,
}

runner = TransformerRunner(model_config, dataset_builder, model_dir='/tmp/transformer')

train_files = [('testdata/train.src.txt','testdata/train.tgt.txt')]
runner.train(train_files, epochs=10, callbacks=None)

```


## BERT的使用


### 从头预训练BERT

如果你想从头开始训练一个预训练的BERT模型：

```python
from transformers_keras import BertForPretrainingModel
from transformers_keras import BertTFRecordDatasetBuilder

model_config = {
    'max_positions': 128,
    'num_layers': 6,
    'vocab_size': 21128,
}

# 构建全新的模型，参数采用一定初始化算法进行初始化
model = BertForPretrainingModel(**model_config)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

model.compile(optimizer='adam', loss=loss, metrics=[metric])

model(model.dummy_inputs())
model.summary()


# 假设你的训练数据已经整理成TFRecord格式
builder = BertTFRecordDatasetBuilder(max_sequence_length=128, record_option='GZIP')
train_files = ['testdata/bert_custom_pretrain.tfrecord']
train_dataset = builder.build_train_dataset(train_files)

model.fit(train_dataset, epochs=2)
```

### 加载预训练好的BERT模型

你也可以直接加载预训练模型。加载预训练模型，有两个选择：

* 加载完整的预训练`BERT`模型，包含`MLM`、`NSP`任务
* 加载`BERT`的`Encoder`部分


#### 加载完整的`BERT`模型

```python
from transformers_keras import BertForPretrainingModel
from transformers_keras import BertTFRecordDatasetBuilder

# 以chinese-bert-base为例，你可以使用其他模型
PRETRAINED_MODEL_PATH = '/path/to/chinese_L-12_H-768_A-12'

# 加载预训练模型
model = BertForPretrainingModel.from_pretrained(PRETRAINED_MODEL_PATH)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')

model.compile(optimizer='adam', loss=loss, metrics=[metric])

model(model.dummy_inputs())
model.summary()

# 使用举例：
#   加载预训练模型之后，继续使用自己的语料训练模型
#   也可以在本模型之上，增加其他层，然后进行在特定任务上的微调。详情请看其他模型的使用方法

# 假设你的训练数据已经整理成TFRecord格式
builder = BertTFRecordDatasetBuilder(
        max_sequence_length=128, record_option='GZIP', eos_token='T',
        train_repeat_count=100, model_dir='models/bert')
train_files = ['testdata/bert_custom_pretrain.tfrecord']
train_dataset = builder.build_train_dataset(train_files)

model.fit(train_dataset, epochs=2)
```

#### 加载`BERT`的`Encoder`部分

```python
from transformers_keras import BertModel
from transformers_keras import BertTFRecordDatasetBuilder

# 以chinese-bert-base为例，你可以使用其他模型
PRETRAINED_MODEL_PATH = '/path/to/chinese_L-12_H-768_A-12'

# 加载预训练模型
model = BertModel.from_pretrained(PRETRAINED_MODEL_PATH)

# 使用举例：
#   可以直接拿来做特征抽取，输入序列，获得sequence_output和pooled_output

input_ids = [0, 1, 2, 3, 4, 5, 6, 0, 0, 0]
input_ids = input_ids + [0] * (512 - len(input_ids))
input_ids = tf.constant(input_ids, dtype=tf.int32, shape=(1, 512))
input_mask = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
input_mask = input_mask + [1] * (512 - len(input_mask))
input_mask = tf.constant(input_mask, dtype=tf.int32, shape=(1, 512))
segment_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
segment_ids = segment_ids + [0] * (512 - len(segment_ids))
segment_ids = tf.constant(segment_ids, dtype=tf.int32, shape=(1, 512))

# call方式
sequence_outputs, pooled_outputs = model(inputs=(input_ids, input_mask, segment_ids))
self.assertEqual([1, 512, 768], sequence_outputs.shape)
self.assertEqual([1, 768], pooled_outputs.shape)
print(sequence_outputs)
print(pooled_outputs)

# predict方式，结果和call方式应该是一致的
(sequence_outputs, pooled_outputs) = model.predict(x=(input_ids, input_mask, segment_ids))
self.assertEqual((1, 512, 768), sequence_outputs.shape)
self.assertEqual((1, 768), pooled_outputs.shape)
print(sequence_outputs)
print(pooled_outputs)
```

## ALBERT

使用方式和 [BERT](#bert的使用) 基本一样。

