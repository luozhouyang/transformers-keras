import argparse
import json
import logging
import os

from .callbacks import SavedModelExporter, TransformerLearningRate
from .datasets import TransformerTextFileDatasetBuilder, TransformerTFRecordDatasetBuilder
from .modeling_transformer import *
from .tokenizers import TransformerDefaultTokenizer

MODEL_CONFIG = {
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'num_attention_heads': 8,
    'hidden_size': 512,
    'ffn_size': 2048,
    'dropout_rate': 0.2,
    'max_positions': 512,
    'source_vocab_size': 0,  # will be updated by tokenizer
    'target_vocab_size': 0,
}

DATA_CONFIG = {
    'model_dir': 'testdata/transformer',
    'files_format': 'txt',  # txt or tfrecord
    'record_option': 'GZIP',  # used for tfrecord files
    'train_src_files': ['testdata/train.src.txt'],
    'train_tgt_files': ['testdata/train.tgt.txt'],
    'valid_src_files': ['testdata/train.src.txt'],
    'valid_tgt_files': ['testdata/train.tgt.txt'],
    'src_max_len': 16,  # used for tfrecord files
    'tgt_max_len': 16,
    'src_vocab_file': 'testdata/vocab_src.txt',
    'tgt_vocab_file': 'testdata/vocab_tgt.txt',
    'train_shuffle_buffer_size': 100,
    'valid_shuffle_buffer_size': 100,
    'pad_token': '<pad>',
    'unk_token': '<unk>',
    'sos_token': '<s>',
    'eos_token': '</s>',
    'train_repeat_count': 100
}


def build_model(config):
    x = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='x')
    y = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='y')
    model = Transformer(TransformerConfig(**config))
    logits, _, _, _ = model(inputs=(x, y))
    probs = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x), name='probs')(logits)

    model = tf.keras.Model(inputs=[x, y], outputs=[probs])

    lr = TransformerLearningRate(MODEL_CONFIG['hidden_size'])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.98, epsilon=1e-9),
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


def build_datasets():
    dataset_builder = None
    train_dataset, valid_dataset = None, None
    if DATA_CONFIG['files_format'] == 'txt':
        src_vocab_file = DATA_CONFIG['src_vocab_file']
        tgt_vocab_file = DATA_CONFIG['tgt_vocab_file']
        src_tokenizer = TransformerDefaultTokenizer(src_vocab_file)
        tgt_tokenizer = TransformerDefaultTokenizer(
            tgt_vocab_file) if src_vocab_file != tgt_vocab_file else src_tokenizer
        dataset_builder = TransformerTextFileDatasetBuilder(src_tokenizer, tgt_tokenizer, **DATA_CONFIG)

        valid_src_files = DATA_CONFIG['valid_src_files']
        valid_tgt_files = DATA_CONFIG['valid_tgt_files']
        logging.info('Build validation dataset from text files:')
        logging.info('  valid src files: {}'.format(valid_src_files))
        logging.info('  valid tgt files: {}'.format(valid_tgt_files))
        valid_dataset = dataset_builder.build_valid_dataset([(x, y) for x, y in zip(valid_src_files, valid_tgt_files)])

        train_src_files = DATA_CONFIG['train_src_files']
        train_tgt_files = DATA_CONFIG['train_tgt_files']
        logging.info('Build training dataset from text files:')
        logging.info('  train src files: {}'.format(train_src_files))
        logging.info('  train tgt files: {}'.format(train_tgt_files))
        train_dataset = dataset_builder.build_train_dataset([(x, y) for x, y in zip(train_src_files, train_tgt_files)])

        MODEL_CONFIG.update({
            'source_vocab_size': src_tokenizer.vocab_size,
            'target_vocab_size': tgt_tokenizer.vocab_size,
        })

    elif DATA_CONFIG['files_format'] == 'tfrecord':
        dataset_builder = TransformerTFRecordDatasetBuilder(DATA_CONFIG['src_max_len'], DATA_CONFIG['tgt_max_len'])

        valid_src_files = DATA_CONFIG['valid_src_files']
        valid_tgt_files = DATA_CONFIG['valid_tgt_files']
        logging.info('Build validation dataset from tfrecord files:')
        logging.info('  valid src files: {}'.format(valid_src_files))
        logging.info('  valid tgt files: {}'.format(valid_tgt_files))
        valid_dataset = dataset_builder.build_valid_dataset([(x, y) for x, y in zip(valid_src_files, valid_tgt_files)])

        train_src_files = DATA_CONFIG['train_src_files']
        train_tgt_files = DATA_CONFIG['train_tgt_files']
        logging.info('Build training dataset from tfrecord files:')
        logging.info('  train src files: {}'.format(train_src_files))
        logging.info('  train tgt files: {}'.format(train_tgt_files))
        train_dataset = dataset_builder.build_train_dataset([(x, y) for x, y in zip(train_src_files, train_tgt_files)])

    else:
        raise ValueError('Invalid argument `files_format`, must be one of [txt, tfrecord].')
    return dataset_builder, train_dataset, valid_dataset


def train():
    model_dir = DATA_CONFIG['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset_builder, train_dataset, valid_dataset = build_datasets()

    transformer = build_model(MODEL_CONFIG)
    logging.info('Build transformer model finished.')
    logging.info('Transformer config is: \n{}'.format(json.dumps(MODEL_CONFIG, indent=2, ensure_ascii=False)))

    transformer.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=DATA_CONFIG.get('epochs', 10),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            tf.keras.callbacks.TensorBoard(os.path.join(model_dir, 'tensorboard')),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_dir,
                verbose=1,
                save_freq=dataset_builder.train_batch_size * DATA_CONFIG.get('ckpt_steps', 10)
            ),
            SavedModelExporter(
                os.path.join(model_dir, 'export'),
                every_steps=DATA_CONFIG.get('export_steps', 10)),
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', type=str, default='train', choices=['train', 'eval', 'predict'])

    args, _ = parser.parse_known_args()

    if 'train' == args.action:
        train()
