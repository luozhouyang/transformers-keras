import unittest

from transformers_keras.datasets import TransformerTextFileDatasetBuilder
from transformers_keras.modeling_transformer import *
from transformers_keras.tokenizers import TransformerDefaultTokenizer
from .modeling_transformer import Transformer


class ModelingTransformerTest(unittest.TestCase):

    def testEncoder(self):
        config = TransformerConfig(num_encoder_layers=2, hidden_size=512, num_attention_heads=8, ffn_size=2048)
        sample_encoder = TransformerEncoder(config)
        inputs = tf.constant([[i for i in range(62)] * 64], shape=(64, 62), dtype=tf.int64)
        output, attn_weights = sample_encoder(inputs=(inputs, None))

        self.assertEqual(output.shape, [64, 62, config.hidden_size])  # (batch_size, input_seq_len, d_model)

        # each layers attention weights
        for v in attn_weights:
            self.assertEqual(v.shape, [64, config.num_attention_heads, 62, 62])

    def testDecoder(self):
        config = TransformerConfig(num_decoder_layers=2, hidden_size=512, num_attention_heads=8, ffn_size=2048)
        sample_decoder = TransformerDecoder(config)
        y = tf.constant(np.random.randint(0, 10, 43), shape=(1, 43), dtype=tf.int64)
        enc_outputs = tf.random.uniform((1, 50, config.hidden_size))
        inputs = (y, enc_outputs, None, None)
        output, self_attn_weights, context_attn_weights = sample_decoder(inputs=inputs)

        self.assertEqual(output.shape, [1, 43, config.hidden_size])
        for attn in self_attn_weights:
            self.assertEqual(attn.shape, [1, config.num_attention_heads, 43, 43])
        for attn in context_attn_weights:
            self.assertEqual(attn.shape, [1, config.num_attention_heads, 43, 50])

    def buildTransformerModel(self, config):
        x = tf.keras.layers.Input(shape=(32,), dtype=tf.int32, name='x')
        y = tf.keras.layers.Input(shape=(32,), dtype=tf.int32, name='y')
        model = Transformer(config)
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

        model_config = TransformerConfig(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_siz=512,
            ffn_size=2048,
            max_positions=512,
            source_vocab_size=src_tokenizer.vocab_size,
            target_vocab_size=tgt_tokenizer.vocab_size,
        )
        model = Transformer(model_config)
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
        model_config = TransformerConfig(
            num_encoder_layers=2,
            num_decoder_layers=2,
            hidden_siz=512,
            ffn_size=2048,
            max_positions=512,
            source_vocab_size=src_tokenizer.vocab_size,
            target_vocab_size=tgt_tokenizer.vocab_size,
        )
        model = self.buildTransformerModel(model_config)
        # print(model.get_config())
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
