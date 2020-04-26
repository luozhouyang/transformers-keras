import collections
import logging
import os

import tensorflow as tf

from transformers_keras.tokenizers import bert_tokenization


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


class TransformerTFRecordGenerator(object):

    def __init__(self, src_vocab_file, tgt_vocab_file=None, share_vocab=False, **kwargs):
        super().__init__()
        self.src_tokenizer = bert_tokenization.FullTokenizer(
            vocab_file=src_vocab_file,
            do_lower_case=kwargs.get('do_lower_case', True),
            split_on_punc=kwargs.get('split_on_punc', True))
        self.src_vocab = self.src_tokenizer.vocab
        self.src_vocab_words = list(self.src_tokenizer.vocab.keys())
        if share_vocab:
            self.tgt_tokenizer = self.src_tokenizer
            self.tgt_vocab = self.src_vocab
            self.tgt_vocab_words = self.src_vocab_words
        else:
            assert tgt_vocab_file is not None, "tgt_vocab_file must not be None if share_vocab=False"
            self.tgt_tokenizer = bert_tokenization.FullTokenizer(
                vocab_file=tgt_vocab_file,
                do_lower_case=kwargs.get('do_lower_case', True),
                split_on_punc=kwargs.get('split_on_punc', True)
            )
            self.tgt_vocab = self.tgt_tokenizer.vocab
            self.tgt_vocab_words = list(self.tgt_vocab.keys())

        self.max_src_sequence_length = kwargs.get('max_src_sequence_length', 128)
        self.max_tgt_sequence_length = kwargs.get('max_tgt_sequence_length', 128)

        self.src_sos_token = kwargs.get('src_sos_token', '<S>')
        self.tgt_sos_token = kwargs.get('tgt_sos_token', '<S>')

        self.src_eos_token = kwargs.get('src_eos_token', '<T>')
        self.tgt_eos_token = kwargs.get('tgt_eos_token', '<T>')

        self.src_pad_token = kwargs.get('src_pad_token', '[PAD]')
        self.tgt_pad_token = kwargs.get('tgt_pad_token', '[PAD]')

        self.src_unk_token = kwargs.get('src_unk_token', '[UNK]')
        self.src_unk_id = self.src_vocab[self.src_unk_token]
        self.tgt_unk_token = kwargs.get('tgt_unk_token', '[UNK]')
        self.tgt_unk_id = self.tgt_vocab[self.tgt_unk_token]

        self.record_option = kwargs.get('record_option', 'GZIP')
        self.log_steps = kwargs.get('log_steps', 1000)

    def generate(self, input_files, output_files):
        writers = []
        for output_file in output_files:
            writers.append(tf.io.TFRecordWriter(output_file, options=self.record_option))

        total = 0
        writer_index = 0
        for input_file in input_files:
            if not os.path.exists(input_file):
                logging.warning('File %s does not exists.' % input_file)
                continue
            with open(input_file, mode='rt', encoding='utf8') as fin:
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    src, tgt = self._parse_line(line)
                    if not src or not tgt:
                        continue

                    src_tokens = self.src_tokenizer.tokenize(src)
                    if not src_tokens or len(src_tokens) > self.max_src_sequence_length - 2:
                        continue
                    tgt_tokens = self.tgt_tokenizer.tokenize(tgt)
                    if not tgt_tokens or len(tgt_tokens) > self.max_tgt_sequence_length - 2:
                        continue

                    src_tokens = [self.src_sos_token] + src_tokens + [self.src_eos_token]
                    tgt_tokens = [self.tgt_sos_token] + tgt_tokens + [self.tgt_eos_token]
                    while len(src_tokens) < self.max_src_sequence_length:
                        src_tokens.append(self.src_pad_token)
                    assert len(src_tokens) == self.max_src_sequence_length

                    while len(tgt_tokens) < self.max_tgt_sequence_length:
                        tgt_tokens.append(self.tgt_pad_token)
                    assert len(tgt_tokens) == self.max_tgt_sequence_length

                    src_ids = [self.src_vocab.get(t, self.src_unk_id) for t in src_tokens]
                    tgt_ids = [self.tgt_vocab.get(t, self.tgt_unk_id) for t in tgt_tokens]

                    instance = {
                        'src_tokens': src_tokens,
                        'src_ids': src_ids,
                        'tgt_tokens': tgt_tokens,
                        'tgt_ids': tgt_ids
                    }

                    try:
                        example = self._create_example(instance)
                        if example is not None:
                            writers[writer_index].write(example.SerializeToString())
                            total += 1
                    except Exception as e:
                        logging.warning('Write example exception: %s' % e)
                        continue

                    if total % self.log_steps == 0:
                        logging.info('Write %d examples.' % total)
                    writer_index = (writer_index+1) % len(writers)

        logging.info('Write %d examples in total.' % total)
        for writer in writers:
            writer.close()

    def _parse_line(self, line):
        parts = line.split('@')
        if len(parts) != 2:
            return '', ''
        src, tgt = parts[0].strip(), parts[1].strip()
        return src, tgt

    def _create_example(self, instance):
        features = collections.OrderedDict()
        features['src_ids'] = create_int_feature(instance['src_ids'])
        features['tgt_ids'] = create_int_feature(instance['tgt_ids'])
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example
