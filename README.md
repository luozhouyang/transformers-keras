# transformers-keras

![Python package](https://github.com/luozhouyang/transformers-keras/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/transformers-keras.svg)](https://badge.fury.io/py/transformers-keras)
[![Python](https://img.shields.io/pypi/pyversions/transformers-keras.svg?style=plastic)](https://badge.fury.io/py/transformers-keras)


基于`tf.keras`的Transformers系列模型实现。

所有的`Model`都是keras模型，可以直接用于训练模型、评估模型或者导出模型用于部署。

在线文档：[transformers-keras文档](https://transformers-keras.readthedocs.io/zh_CN/latest/index.html)

本库功能预览：

* 加载各种预训练模型的权重
* 句向量的解决方案
* 抽取式问答任务的解决方案
* 序列分类任务的解决方案
* 序列标注任务的解决方案

以上任务的数据处理、模型构建都完成了，可以一键开始训练。这里有两个例子：

* [使用BERT微调文本分类任务](https://transformers-keras.readthedocs.io/zh_CN/latest/start.html#id4)
* [使用BERT微调问答任务](https://transformers-keras.readthedocs.io/zh_CN/latest/start.html#id5)


更多使用方法和介绍，请查看在线文档：[transformers-keras文档](https://transformers-keras.readthedocs.io/zh_CN/latest/index.html)