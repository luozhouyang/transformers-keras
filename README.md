# transformers-keras

![Python package](https://github.com/luozhouyang/transformers-keras/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/transformers-keras.svg)](https://badge.fury.io/py/transformers-keras)
[![Python](https://img.shields.io/pypi/pyversions/transformers-keras.svg?style=plastic)](https://badge.fury.io/py/transformers-keras)

[English](README_EN.md) | 中文文档 

基于`tf.keras`的Transformers系列模型实现。

所有的`Model`都是keras模型，可以直接用于训练模型、评估模型或者导出模型用于部署。

# 目录
1. [安装](#安装)
2. [实现的模型](#实现的模型)
3. [BERT](#BERT)
    - 3.1 [BERT支持的预训练权重](#BERT)
    - 3.2 [BERT特征抽取示例](#BERT特征抽取示例)
    - 3.3 [BERT微调模型示例](#BERT微调模型示例)
        - 3.3.1 [使用函数式API构建BERT微调模型](#使用函数式API构建BERT微调模型)
        - 3.3.2 [使用预制的BertForSequenceClassification](#使用预制的BertForSequenceClassification)
        - 3.3.3 [使用预制的BertForQuestionAnswering](#使用预制的BertForQuestionAnswering)
    - 3.4 [BERT模型导出和部署](#BERT导出SavedModel格式的模型用Serving部署)
4. [ALBERT](#ALBERT)
    - 4.1 [ALBERT支持的预训练权重](#ALBERT)
    - 4.2 [ALBERT特征抽取示例](#ALBERT特征抽取示例)
    - 4.3 [ALBERT微调模型示例](#ALBERT微调模型示例)
        - 4.3.1 [使用函数式API构建ALBERT微调模型](#使用函数式API构建ALBERT微调模型)
        - 4.3.2 [使用预制的AlbertForSequenceClassification](#使用预制的AlbertForSequenceClassification)
        - 4.3.3 [使用预制的AlbertForQuestionAnswering](#使用预制的AlbertForQuestionAnswering)
    - 4.4 [ALBERT模型导出和部署](#ALBERT导出SavedModel格式的模型用Serving部署)
5. [进阶使用](#进阶使用)
    - 5.1 [加载时跳过一些参数的权重](#加载预训练模型权重的过程中跳过一些参数的权重)
    - 5.2 [加载第三方模型实现的权重](#加载第三方实现的模型的权重)


## 安装

```bash
pip install -U transformers-keras
```

## 实现的模型

- [x] Transformer[*已删除*]
  * [Attention Is All You Need](https://arxiv.org/abs/1706.03762). 
  * TensorFlow官方教程:[Transformer model for language understanding](https://www.tensorflow.org/beta/tutorials/text/transformer)
- [x] BERT
  * [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- [x] ALBERT
  * [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)


## BERT

支持加载的预训练BERT模型权重:

* 所有使用 [google-research/bert](https://github.com/google-research/bert) 训练的**BERT**模型
* 所有使用 [ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) 训练的**BERT**和**RoBERTa**模型

### BERT特征抽取示例

```python
from transformers_keras import Bert

# 加载预训练模型权重
model = Bert.from_pretrained('/path/to/pretrained/bert/model')
input_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
sequence_outputs, pooled_output = model(inputs=[input_ids, segmet_ids, attention_mask], training=False)

```

另外，可以通过构造器参数 `return_states=True` 和 `return_attention_weights=True` 来获取每一层的 `hidden_states` 和 `attention_weights` 输出:

```python
from transformers_keras import Bert

# 加载预训练模型权重
model = Bert.from_pretrained(
    '/path/to/pretrained/bert/model', 
    return_states=True, 
    return_attention_weights=True)
input_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
sequence_outputs, pooled_output, hidden_states, attn_weights = model(
    inputs=[input_ids, segment_ids, attention_mask], training=False)

```

### BERT微调模型示例

微调模型有两种风格，一种是使用`keras functional api`构建新的模型，一种是直接`subclassing Model`的方式。

这里提供一个使用`functional api`构建微调模型的例子：

* [使用函数式API构建BERT微调模型](#使用函数式API构建BERT微调模型)

这里提供两个预制的微调模型，都是使用`subclassing`的方式实现的：

* [使用预制的BertForSequenceClassification](#使用预制的BertForSequenceClassification)
* [使用预制的BertForQuestionAnswering](#使用预制的BertForQuestionAnswering)


#### 使用函数式API构建BERT微调模型

以构建一个序列分类网络为例：

```python
from transformers_keras import Bert

input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
bert = Bert.from_pretrained('/path/to/pretrained/model')
_, pooled_output = bert(inputs=[input_ids, segment_ids, attention_mask])
outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
model = tf.keras.Model(inputs=[input_ids, segment_ids, attention_mask], outputs=outputs)
model.compile(loss='binary_cross_entropy', optimizer='adam')
model.summary()
```

可以得到以下的网络输出：
```bash
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (Bert)                     [(None, None, 768),  102267648   input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            1538        bert[0][1]                       
==================================================================================================
Total params: 102,269,186
Trainable params: 102,269,186
Non-trainable params: 0
__________________________________________________________________________________________________
```

#### 使用预制的BertForSequenceClassification

你可以使用BERT构建序列的二分类网络：

```python
from transformers_keras import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('/path/to/pretrained/model')
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
model.fit(train_dataset, epoch=10, callbacks=[])
```

可以得到下面的模型输出：
```bash
Model: "bert_for_sequence_classification"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (BertModel)                ((None, None, 768),  59740416    input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            1538        bert[0][1]                       
==================================================================================================
Total params: 59,741,954
Trainable params: 59,741,954
Non-trainable params: 0
__________________________________________________________________________________________________
```

#### 使用预制的BertForQuestionAnswering

另一个例子，使用BERT来做Question Answering：

```python
from transformers_keras import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('/path/to/pretrained/model')
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
model.fit(train_dataset, epoch=10, callbacks=[])
```

可以得到下面的模型输出：
```bash
Model: "bert_for_question_answering"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
bert (BertModel)                ((None, None, 768),  59740416    input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, None, 2)      1538        bert[0][0]                       
__________________________________________________________________________________________________
head (Lambda)                   (None, None)         0           dense[0][0]                      
__________________________________________________________________________________________________
tail (Lambda)                   (None, None)         0           dense[0][0]                      
==================================================================================================
Total params: 59,741,954
Trainable params: 59,741,954
Non-trainable params: 0
__________________________________________________________________________________________________
```

### BERT导出SavedModel格式的模型用Serving部署

你可以很方便地把模型转换成SavedModel格式。

这里是直接把BERT模型导出部署的示例:

```python
# 加载预训练模型权重
model = Bert.from_pretrained(
    '/path/to/pretrained/bert/model', 
    return_states=True, 
    return_attention_weights=True)
model.save('/path/to/save')
```

接下来，就可以使用 [tensorflow/serving](https://github.com/tensorflow/serving) 来部署模型了。

> 本项目所有的模型，都可以使用这种方式导出成SavedModel格式，然后直接用serving部署。


## ALBERT

支持加载的预训练ALBERT模型权重:

* 所有使用 [google-research/albert](https://github.com/google-research/albert) 训练的模型。

### ALBERT特征抽取示例

```python
from transformers_keras import Albert

# 加载预训练权重
model = Albert.from_pretrained('/path/to/pretrained/albert/model')
input_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
sequence_outputs, pooled_output = model(inputs=[input_ids, segment_ids, attention_mask], training=False)
```

另外，可以通过构造器参数 `return_states=True` 和 `return_attention_weights=True` 来获取每一层的 `hidden_states` 和 `attention_weights` 输出:

```python
from transformers_keras import Albert

# 加载预训练模型权重
model = Albert.from_pretrained(
    '/path/to/pretrained/albert/model', 
    return_states=True, 
    return_attention_weights=True)

input_ids = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
segment_ids = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
attention_mask = tf.constant([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
sequence_outputs, pooled_output, states, attn_weights = model(
    inputs=[input_ids, segment_ids, attention_mask], training=False)
```

### ALBERT微调模型示例


微调模型有两种风格，一种是使用`keras functional api`构建新的模型，一种是直接`subclassing Model`的方式。

这里提供一个使用`functional api`构建微调模型的例子：

* [使用函数式API构建ALBERT微调模型](#使用函数式API构建ALBERT微调模型)

这里提供两个预制的微调模型，都是使用`subclassing`的方式实现的：

* [使用预制的AlbertForSequenceClassification](#使用预制的AlbertForSequenceClassification)
* [使用预制的AlbertForQuestionAnswering](#使用预制的AlbertForQuestionAnswering)

#### 使用函数式API构建ALBERT微调模型

以构建一个序列分类网络为例：

```python
input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='segment_ids')
attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')
albert = Albert.from_pretrained(os.path.join(BASE_DIR, 'albert_base_zh'))
albert.trainable = trainable
_, pooled_output = albert(inputs=[input_ids, segment_ids, attention_mask])
outputs = tf.keras.layers.Dense(2, name='output')(pooled_output)
model = tf.keras.Model(inputs=[input_ids, segment_ids, attention_mask], outputs=outputs)
model.compile(loss='binary_cross_entropy', optimizer='adam')

model.summary()
```

可以得到以下网络输出：

````bash
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
albert (Albert)                 [(None, None, 768),  10547968    input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
output (Dense)                  (None, 2)            1538        albert[0][1]                     
==================================================================================================
Total params: 10,549,506
Trainable params: 10,549,506
Non-trainable params: 0
__________________________________________________________________________________________________

```


#### 使用预制的AlbertForSequenceClassification

你可以使用ALBERT构建序列的二分类网络：

```python
from transformers_keras import AlbertForSequenceClassification

model = AlbertForSequenceClassification.from_pretrained('/path/to/pretrained/model')
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
model.fit(train_dataset, epoch=10, callbacks=[])

```

可以得到下面的模型输出：

```bash
Model: "albert_for_sequence_classification"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
albert (AlbertModel)            ((None, None, 768),  10547968    input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, 2)            1538        albert[0][1]                     
==================================================================================================
Total params: 10,549,506
Trainable params: 10,549,506
Non-trainable params: 0
__________________________________________________________________________________________________

```

#### 使用预制的AlbertForQuestionAnswering

另一个例子，使用BERT来做Question Answering：

```python
from transformers_keras import AlbertForQuestionAnswering

model = AlbertForQuestionAnswering.from_pretrained('/path/to/pretrained/model')
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['acc']
)
model.fit(train_dataset, epoch=10, callbacks=[])
```

可以得到下面的模型输出：

```bash
Model: "albert_for_question_answering"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_ids (InputLayer)          [(None, None)]       0                                            
__________________________________________________________________________________________________
segment_ids (InputLayer)        [(None, None)]       0                                            
__________________________________________________________________________________________________
attention_mask (InputLayer)     [(None, None)]       0                                            
__________________________________________________________________________________________________
albert (AlbertModel)            ((None, None, 768),  10547968    input_ids[0][0]                  
                                                                 segment_ids[0][0]                
                                                                 attention_mask[0][0]             
__________________________________________________________________________________________________
dense (Dense)                   (None, None, 2)      1538        albert[0][0]                     
__________________________________________________________________________________________________
head (Lambda)                   (None, None)         0           dense[0][0]                      
__________________________________________________________________________________________________
tail (Lambda)                   (None, None)         0           dense[0][0]                      
==================================================================================================
Total params: 10,549,506
Trainable params: 10,549,506
Non-trainable params: 0
__________________________________________________________________________________________________
```


### ALBERT导出SavedModel格式的模型用Serving部署

你可以很方便地把模型转换成SavedModel格式。这里是一个示例:

```python
# 加载预训练模型权重
model = Albert.from_pretrained(
    '/path/to/pretrained/albert/model', 
    return_states=True, 
    return_attention_weights=True)
model.save('/path/to/save')
```

接下来，就可以使用 [tensorflow/serving](https://github.com/tensorflow/serving) 来部署模型了。

> 本项目所有的模型，都可以使用这种方式导出成SavedModel格式，然后直接用serving部署。


## 进阶使用

支持的高级使用方法:

* 加载预训练模型权重的过程中跳过一些参数的权重
* 加载第三方实现的模型的权重

### 加载预训练模型权重的过程中跳过一些参数的权重

有些情况下，你可能会在加载预训练权重的过程中，跳过一些权重的加载。这个过程很简单。

这里是一个示例：

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

所有支持跳过加载的权重如下:

* `skip_token_embedding`, 跳过加载ckpt的 `token_embedding` 权重
* `skip_position_embedding`, 跳过加载ckpt的 `position_embedding` 权重
* `skip_segment_embedding`, 跳过加载ckpt的 `token_type_emebdding` 权重
* `skip_embedding_layernorm`, 跳过加载ckpt的 `layer_norm` 权重
* `skip_pooler`, 跳过加载ckpt的 `pooler` 权重



### 加载第三方实现的模型的权重

在有一些情况下，第三方实现了一些模型，它的权重的结构组织和官方的实现不太一样。对于一般的预训练加载库，实现这个功能是需要库本身修改代码来实现的。本库通过 **适配器模式** 提供了这种支持。用户只需要继承 **AbstractAdapter** 即可实现自定义的权重加载逻辑。

```python
from transformers_keras.adapters import AbstractAdapter
from transformers_keras import Bert, Albert

# 自定义的BERT权重适配器
class MyBertAdapter(AbstractAdapter):

    def adapte_config(self, config_file, **kwargs):
        # 在这里把配置文件的配置项，转化成本库的BERT需要的配置
        # 本库实现的BERT所需参数都在构造器里，可以简单方便得查看
        pass

    def adapte_weights(self, model, config, ckpt, **kwargs):
        # 在这里把ckpt的权重设置到model的权重里
        # 可以参考BertAdapter的实现过程
        pass

# 加载预训练权重的时候，指定自己的适配器 `adapter=MyBertAdapter()`
bert = Bert.from_pretrained('/path/to/your/bert/model', adapter=MyBertAdapter())

# 自定义的ALBERT权重适配器
class MyAlbertAdapter(AbstractAdapter):

    def adapte_config(self, config_file, **kwargs):
        # 在这里把配置文件的配置项，转化成本库的BERT需要的配置
        # 本库实现的ALBERT所需参数都在构造器里，可以简单方便得查看
        pass

    def adapte_weights(self, model, config, ckpt, **kwargs):
        # 在这里把ckpt的权重设置到model的权重里
        # 可以参考AlbertAdapter的实现过程
        pass

# 加载预训练权重的时候，指定自己的适配器 `adapter=MyAlbertAdapter()`
albert = Albert.from_pretrained('/path/to/your/albert/model', adapter=MyAlbertAdapter())
```
