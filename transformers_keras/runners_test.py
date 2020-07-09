import tensorflow as tf

from transformers_keras import (
    BertTFRecordDatasetBuilder,
    TransformerDefaultTokenizer,
    TransformerRunner,
    TransformerTextFileDatasetBuilder,
)


class RunnersTest(tf.test.TestCase):

    def testTransformer(self):

        src_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_src.txt')
        tgt_tokenizer = TransformerDefaultTokenizer(vocab_file='testdata/vocab_tgt.txt')
        dataset_builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer, train_repeat_count=100)

        model_config = {
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'src_vocab_size': src_tokenizer.vocab_size,
            'tgt_vocab_size': tgt_tokenizer.vocab_size,
        }

        runner = TransformerRunner(model_config, dataset_builder, model_dir='models/transformer')

        train_files = [('testdata/train.src.txt', 'testdata/train.tgt.txt')]
        runner.train(train_files, epochs=10, callbacks=None)


if __name__ == "__main__":
    tf.test.main()
