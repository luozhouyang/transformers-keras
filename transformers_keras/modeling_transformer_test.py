import unittest

import tensorflow as tf

from transformers_keras.datasets import TransformerTextFileDatasetBuilder
from transformers_keras.modeling_transformer import *
from transformers_keras.tokenizers import TransformerDefaultTokenizer

from .modeling_transformer import Transformer


class ModelingTransformerTest(tf.test.TestCase):

    def testPositionalEncoding(self):
        pe = PositionalEncoding(512, 512)
        x = np.random.randint(0, 512, size=(2, 16))
        x = tf.constant(x, shape=(2, 16), dtype=tf.int64)

        encoding = pe(x)
        self.assertEqual([1, 16, 512], encoding.shape)  # shape[0] will be broadcasted when add token embedding

        y = np.random.randint(0, 512, size=(2, 16))
        y = tf.constant(y, shape=(2, 16), dtype=tf.int64)
        encoding2 = pe(y)
        self.assertEqual([1, 16, 512], encoding2.shape)
        self.assertAllEqual(encoding.numpy(), encoding2.numpy())

    def testEmbedding(self):
        te = TransformerEmbedding(100, 128, 128)
        x = np.random.randint(0, 100, size=(2, 16))
        x = tf.constant(x, shape=(2, 16), dtype=tf.int64)
        embedding = te(x)
        self.assertEqual([2, 16, 128], embedding.shape)

        y = x[:, :]
        embedding2 = te(y)
        self.assertEqual([2, 16, 128], embedding2.shape)
        self.assertAllEqual(embedding, embedding2)

    def testEncoder(self):
        encoder = TransformerEncoder(vocab_size=100, num_layers=2)
        inputs = tf.constant([[i for i in range(10)] * 16], shape=(16, 10), dtype=tf.int64)
        output, attn_weights = encoder(inputs=(inputs, None))

        self.assertEqual(output.shape, [16, 10, encoder.hidden_size])  # (batch_size, input_seq_len, d_model)

        # each layers attention weights
        for v in attn_weights:
            self.assertEqual(v.shape, [16, encoder.num_attention_heads, 10, 10])

    def testDecoder(self):
        decoder = TransformerDecoder(vocab_size=100, num_layers=2)
        y = tf.constant(np.random.randint(0, 10, size=(2, 43)), shape=(2, 43), dtype=tf.int64)
        enc_outputs = tf.random.uniform((2, 50, decoder.hidden_size))
        inputs = (y, enc_outputs, None, None)
        output, self_attn_weights, context_attn_weights = decoder(inputs=inputs)

        self.assertEqual(output.shape, [2, 43, decoder.hidden_size])
        for attn in self_attn_weights:
            self.assertEqual(attn.shape, [2, decoder.num_attention_heads, 43, 43])
        for attn in context_attn_weights:
            self.assertEqual(attn.shape, [2, decoder.num_attention_heads, 43, 50])

    def buildTransformerModel(self, config):
        x = tf.keras.layers.Input(shape=(32,), dtype=tf.int32, name='x')
        y = tf.keras.layers.Input(shape=(32,), dtype=tf.int32, name='y')
        model = Transformer(
            src_vocab_size=config.get('src_vocab_size', 100),
            tgt_vocab_size=config.get('tgt_vocab_size', 100),
            max_positions=config.get('max_positions', 512),
            hidden_size=config.get('hidden_size', 512),
            num_encoder_layers=config.get('num_encoder_layers', 2),
            num_decoder_layers=config.get('num_decoder_layers', 2),
            num_attention_heads=config.get('num_attention_heads', 8),
            ffn_size=config.get('ffn_size', 2048),
            dropout_rate=config.get('dropout_rate', 0.2),
            epsilon=config.get('epsilon', 1e-6)
        )
        logits, _, _, _ = model(inputs=(x, y))
        probs = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='probs')(logits)

        model = tf.keras.Model(inputs=[x, y], outputs=[probs])

        model.compile(
            optimizer='adam',  # TODO(zhouyang.luo) schedule learning rate
            loss={
                'probs': tf.keras.losses.SparseCategoricalCrossentropy(name='loss', from_logits=False),
            },
            metrics={
                'probs': [
                    tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                ]
            }
        )

        model.summary()
        return model

    def buildTokenizers(self):
        src_tokenizer = TransformerDefaultTokenizer(
            vocab_file='testdata/vocab_src.txt',
            pad_token='<pad>', unk_token='<unk>', sos_token='<s>', eos_token='</s>'
        )
        tgt_tokenizer = TransformerDefaultTokenizer(
            vocab_file='testdata/vocab_tgt.txt',
            pad_token='<pad>', unk_token='<unk>', sos_token='<s>', eos_token='</s>'
        )
        return src_tokenizer, tgt_tokenizer

    def testTransformerModel(self):
        src_tokenizer, tgt_tokenizer = self.buildTokenizers()
        data_config = {
            'train_shuffle_buffer_size': 100,
            'valid_shuffle_buffer_size': 100,
            'train_repeat_count': 100,
            'train_batch_size': 2,
            'valid_batch_size': 2,
        }
        dataset_builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer, **data_config)

        model = Transformer(
            src_vocab_size=src_tokenizer.vocab_size,
            tgt_vocab_size=tgt_tokenizer.vocab_size,
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        train_dataset = dataset_builder.build_train_dataset(
            train_files=[('testdata/train.src.txt', 'testdata/train.tgt.txt')])
        train_dataset = train_dataset.map(lambda x, y: (x[0], x[1]))
        print(next(iter(train_dataset)))

        for i, v in enumerate(train_dataset):
            outputs, _, _, _ = model(v)

    def testTraining(self):
        src_tokenizer = TransformerDefaultTokenizer(
            vocab_file='testdata/vocab_src.txt',
            pad_token='<pad>', unk_token='<unk>', sos_token='<s>', eos_token='</s>'
        )
        tgt_tokenizer = TransformerDefaultTokenizer(
            vocab_file='testdata/vocab_tgt.txt',
            pad_token='<pad>', unk_token='<unk>', sos_token='<s>', eos_token='</s>'
        )
        data_config = {
            'train_shuffle_buffer_size': 100,
            'valid_shuffle_buffer_size': 100,
            'train_repeat_count': 100,
            'train_batch_size': 2,
            'valid_batch_size': 2,
        }
        dataset_builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer, **data_config)
        model_config = {
            'src_vocab_size': src_tokenizer.vocab_size,
            'tgt_vocab_size': tgt_tokenizer.vocab_size,
        }
        model = self.buildTransformerModel(model_config)

        train_dataset = dataset_builder.build_train_dataset(
            train_files=[('testdata/train.src.txt', 'testdata/train.tgt.txt')])
        print(next(iter(train_dataset)))
        valid_dataset = dataset_builder.build_valid_dataset(
            valid_files=[('testdata/train.src.txt', 'testdata/train.tgt.txt')])
        model.fit(
            train_dataset,
            validation_data=valid_dataset
        )


if __name__ == "__main__":
    unittest.main()
