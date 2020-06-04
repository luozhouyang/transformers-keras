import collections
import logging
import os

import tensorflow as tf

from transformers_keras.tokenizers import TransformerDefaultTokenizer


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


class TransformerTFRecordGenerator(object):

    def __init__(self,
                 src_tokenizer,
                 tgt_tokenizer,
                 src_max_len=512,
                 tgt_max_len=512,
                 record_option='GZIP',
                 log_steps=1000,
                 **kwargs):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        self.record_option = record_option
        self.log_steps = log_steps

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

                    src_ids = self.src_tokenizer.encode(src)
                    src_ids = [self.src_tokenizer.sos_id] + src_ids + [self.src_tokenizer.eos_id]
                    while len(src_ids) < self.src_max_len:
                        src_ids.append(self.src_tokenizer.pad_id)

                    if len(src_ids) > self.src_max_len:
                        logging.warning('Length of sequence is {}, greater than {}', len(src_ids), self.src_max_len)
                        continue

                    tgt_ids = self.tgt_tokenizer.encode(tgt)
                    tgt_ids = [self.tgt_tokenizer.sos_id] + tgt_ids + [self.tgt_tokenizer.eos_id]
                    while len(tgt_ids) < self.tgt_max_len:
                        tgt_ids.append(self.tgt_tokenizer.pad_id)
                    if len(tgt_ids) > self.tgt_max_len:
                        logging.warning('Length of sequence is {}, greater than {}', len(tgt_ids), self.tgt_max_len)
                        continue

                    instance = {
                        # 'src_tokens': src_tokens,
                        'src_ids': src_ids,
                        # 'tgt_tokens': tgt_tokens,
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
