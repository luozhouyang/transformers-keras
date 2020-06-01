import argparse
import collections
import logging
import os
import random

import tensorflow as tf

from transformers_keras.tokenizers import BertDefaultTokenizer


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


class CustomBertGenerator(object):
    """Generate tfrecord files for training, it's different from original bert generation.
        If you need to use standard bert generation, use `bert_create_pretraining_data.py`
    """

    def __init__(self, vocab_file, **kwargs):
        super().__init__()
        self.do_lower_case = kwargs.pop('do_lower_case', True)
        self.split_on_punc = kwargs.pop('split_on_punc', True)
        self.tokenizer = BertDefaultTokenizer(vocab_file=vocab_file)
        self.rng = random.Random(12345)
        self.max_sequence_length = kwargs.pop('max_sequence_length', 512)
        self.do_whole_word_mask = kwargs.pop('do_whole_word_mask', True)
        self.masked_lm_prob = kwargs.pop('masked_lm_prob', 0.15)
        self.max_predictions_per_seq = kwargs.pop('max_predictions_per_seq', 20)
        self.vocab_words = self.tokenizer.vocab
        self.record_option = kwargs.pop('record_option', None)

    def _parse_segment_strs(self, line):
        line = line.strip('\n')
        strs = line.split('@@@')
        if len(strs) != 5:
            return '', '', -1
        label, jdid, cvid, stra, strb = strs[0], strs[1], strs[2], strs[3], strs[4]
        return stra, strb, int(label)

    def _compose_segments(self, stra, strb):
        tokens_a = self.tokenizer.tokenize(stra)
        tokens_b = self.tokenizer.tokenize(strb)
        self._truncate_seq_pair(tokens_a, tokens_b, self.max_sequence_length - 3, self.rng)
        return tokens_a, tokens_b

    def _masked_lm_process(self, instance):
        tokens = instance['original_tokens']
        outputs = self._create_masked_lm_predictions(
            tokens=tokens,
            masked_lm_prob=self.masked_lm_prob,
            max_predictions_per_seq=self.max_predictions_per_seq,
            vocab_words=self.vocab_words,
            rng=self.rng)

        # len(masked_lm_masked_tokens) == len(tokens)
        # len(masked_lm_masked_positions) == self.max_predictions_per_seq
        # len(masked_lm_origin_tokens) == self.max_predictions_per_seq
        masked_lm_masked_tokens, masked_lm_masked_positions, masked_lm_origin_tokens = outputs
        instance.update({
            'original_ids': self.tokenizer.encode(tokens),  # length == len(tokens)
            'masked_tokens': masked_lm_masked_tokens,  # lenght == len(tokens)
            # length == len(tokens)
            'masked_ids': self.tokenizer.encode(masked_lm_masked_tokens),
            'masked_lm_positions': masked_lm_masked_positions,  # length = max_predictions_per_seq
            'masked_lm_tokens': masked_lm_origin_tokens,  # length = max_predictions_per_seq
            'masked_lm_ids': self.tokenizer.encode(masked_lm_origin_tokens)
        })
        return instance

    def _create_example(self, instance):
        # instance = {
        #     'original_tokens': [], # original tokens
        #     'original_ids': [], # original ids
        #     'masked_tokens': [], # tokens contains [MASK]
        #     'masked_ids': [], # ids contains mask_id
        #     'nsp_label': 0, # nsp label
        #     'masked_lm_positions': [], # positions of masked tokens in sequence
        #     'masked_lm_tokens': [], # original tokens replaced by [MASK]
        #     'masked_lm_ids': [], # original tokens' ids replaced by [MASK]
        # }
        input_ids = instance['masked_ids']
        segment_ids = instance['segment_ids']
        original_ids = instance['original_ids']
        assert len(input_ids) == len(segment_ids)
        assert len(input_ids) == len(original_ids)

        input_mask = [1] * len(input_ids)  # masking padding positions
        while len(input_ids) < self.max_sequence_length:
            input_ids.append(self.pad_id)
            segment_ids.append(self.pad_id)
            input_mask.append(0)
            original_ids.append(0)

        assert len(input_ids) == self.max_sequence_length
        assert len(segment_ids) == self.max_sequence_length
        assert len(input_mask) == self.max_sequence_length
        assert len(original_ids) == self.max_sequence_length

        masked_lm_positions = instance['masked_lm_positions']
        masked_lm_ids = instance['masked_lm_ids']
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        while len(masked_lm_positions) < self.max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        assert len(masked_lm_positions) == self.max_predictions_per_seq
        assert len(masked_lm_ids) == self.max_predictions_per_seq
        assert len(masked_lm_weights) == self.max_predictions_per_seq

        features = collections.OrderedDict()
        features['original_ids'] = create_int_feature(instance['original_ids'])  # original ids of tokens
        features['input_ids'] = create_int_feature(input_ids)  # contains masked tokens
        features['input_mask'] = create_int_feature(input_mask)
        features['segment_ids'] = create_int_feature(segment_ids)
        features['next_sentence_labels'] = create_int_feature([instance['next_sentence_labels']])
        features['masked_lm_positions'] = create_int_feature(masked_lm_positions)
        features['masked_lm_ids'] = create_int_feature(masked_lm_ids)
        features['masked_lm_weights'] = create_float_feature(masked_lm_weights)

        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example

    def generate(self, input_files, output_files):
        """Convert pretrain corpus to tfrecord files. Each line of input_files should contains two segments string.
            e.g aaaaaaa@bbbbbb
        """
        writers = []
        for output_file in output_files:
            writers.append(tf.io.TFRecordWriter(output_file, options=self.record_option))

        writer_index = 0
        total = 0
        for input_file in input_files:
            if not os.path.exists(input_file):
                logging.warning('Input file: %s does not exists.' % input_file)
                continue
            with open(input_file, mode='rt', encoding='utf8') as fin:
                while True:
                    line = fin.readline()
                    if not line:
                        break
                    stra, strb, nsp_label = self._parse_segment_strs(line)
                    if not stra or not strb or nsp_label < 0:
                        continue

                    tokens_a, tokens_b = self._compose_segments(stra, strb)
                    if not tokens_a or not tokens_b:
                        continue

                    tokens, segment_ids = ['[CLS]'], [0]
                    for t in tokens_a:
                        tokens.append(t)
                        segment_ids.append(0)
                    tokens.append('[SEP]')
                    segment_ids.append(0)
                    for t in tokens_b:
                        tokens.append(t)
                        segment_ids.append(1)
                    tokens.append('[SEP]')
                    segment_ids.append(1)

                    assert len(tokens) == len(segment_ids)

                    instance = {
                        'original_tokens': tokens,
                        'segment_ids': segment_ids,
                        'next_sentence_labels': nsp_label,
                    }

                    instance = self._masked_lm_process(instance)

                    try:
                        example = self._create_example(instance)
                        if example is not None:
                            writers[writer_index].write(example.SerializeToString())
                            total += 1
                    except Exception as e:
                        logging.warning('Create example exception: %s' % e)
                        continue

                    if total % 10000 == 0:
                        logging.info('Write %d examples.' % total)

                    # updated writer index
                    writer_index = (writer_index + 1) % len(writers)

        for writer in writers:
            writer.close()

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_num_tokens, rng):
        """Truncates a pair of sequences to a maximum sequence length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_num_tokens:
                break

            trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
            assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if rng.random() < 0.5:
                del trunc_tokens[0]
            else:
                trunc_tokens.pop()

    def _create_masked_lm_predictions(self, tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if (self.do_whole_word_mask and len(cand_indexes) >= 1 and
                    token.startswith("##")):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        # cand_indexes: [[1],[2,3],[4],[5],[6,7,8]]
        rng.shuffle(cand_indexes)
        # cand_indexes: [[5],[2,3],[1],[6,7,8],[4]]

        masked_lm_masking_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None
                # 80% of the time, replace with [MASK]
                if rng.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

                masked_lm_masking_tokens[index] = masked_token

                masked_lms.append((index, tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x[0])

        masked_lm_masking_positions = []
        masked_lm_original_tokens = []
        for p in masked_lms:
            masked_lm_masking_positions.append(p[0])
            masked_lm_original_tokens.append(p[1])

        return (masked_lm_masking_tokens, masked_lm_masking_positions, masked_lm_original_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="""Comma splited file path""")
    parser.add_argument('--output_file', type=str, help="""Comma splited file path""")
    parser.add_argument('--vocab_file', type=str, help="""Vocab file""")
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--split_on_punc', type=bool, default=True)
    parser.add_argument('--max_sequence_length', type=int, default=512)
    parser.add_argument('--max_predictions_per_seq', type=int, default=20)
    parser.add_argument('--do_whole_word_mask', type=bool, default=True)
    parser.add_argument('--masked_lm_prob', type=float, default=0.15)
    parser.add_argument('--unk_token', type=str, default='[UNK]')
    parser.add_argument('--pad_token', type=str, default='[PAD]')
    parser.add_argument('--record_option', type=str, default=None)

    args, _ = parser.parse_known_args()

    input_files = str(args.input_file).split(',')
    output_files = str(args.output_file).split(',')

    config = {
        'do_lower_case': args.do_lower_case,
        'split_on_punc': args.split_on_punc,
        'max_sequence_length': args.max_sequence_length,
        'max_predictions_per_seq': args.max_predictions_per_seq,
        'do_whole_word_mask': args.do_whole_word_mask,
        'masked_lm_prob': args.masked_lm_prob,
        'unk_token': args.unk_token,
        'pad_token': args.pad_token,
        'record_option': args.record_option
    }
    g = CustomBertGenerator(args.vocab_file, **config)
    g.generate(input_files, output_files)
