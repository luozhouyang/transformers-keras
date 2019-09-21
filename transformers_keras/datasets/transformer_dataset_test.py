import tensorflow as tf

from transformers_keras.tokenizers.space_tokenizer import SpaceTokenizer
from transformers_keras.datasets.transformer_dataset import TransformerDataset


class TransformerDatasetTest(tf.test.TestCase):

    def testBuildTrainDataset(self):
        src_corpus_files = ['testdata/train.src.txt']
        tgt_corpus_files = ['testdata/train.tgt.txt']
        src_tokenizer = SpaceTokenizer()
        tgt_tokenizer = SpaceTokenizer()
        src_tokenizer.build_from_corpus(src_corpus_files)
        tgt_tokenizer.build_from_corpus(tgt_corpus_files)
        print('src vocab size: ', src_tokenizer.vocab_size)
        print('tgt vocab size: ', tgt_tokenizer.vocab_size)
        d = TransformerDataset(src_tokenizer, tgt_tokenizer, None)

        train_dataset = d.build_train_dataset(train_files=(src_corpus_files, tgt_corpus_files))

        batch = next(iter(train_dataset))
        print(batch)

    def testBuildEvalDataset(self):
        src_corpus_files = ['testdata/train.src.txt']
        tgt_corpus_files = ['testdata/train.tgt.txt']
        src_tokenizer = SpaceTokenizer()
        tgt_tokenizer = SpaceTokenizer()
        src_tokenizer.build_from_corpus(src_corpus_files)
        tgt_tokenizer.build_from_corpus(tgt_corpus_files)
        print('src vocab size: ', src_tokenizer.vocab_size)
        print('tgt vocab size: ', tgt_tokenizer.vocab_size)
        d = TransformerDataset(src_tokenizer, tgt_tokenizer, None)

        eval_dataset = d.build_eval_dataset(eval_files=(src_corpus_files, tgt_corpus_files))

        batch = next(iter(eval_dataset))
        print(batch)

    def testBuildPredictDataset(self):
        src_corpus_files = ['testdata/train.src.txt']
        src_tokenizer = SpaceTokenizer()
        src_tokenizer.build_from_corpus(src_corpus_files)
        print('src vocab size: ', src_tokenizer.vocab_size)
        d = TransformerDataset(src_tokenizer, None, None)

        predict_dataset = d.build_predict_dataset(predict_files=src_corpus_files)

        batch = next(iter(predict_dataset))
        print(batch)


if __name__ == '__main__':
    tf.test.main()
