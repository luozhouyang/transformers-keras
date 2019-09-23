from transformers_keras.datasets.dataset import Dataset
import tensorflow as tf


class TransformerDataset(Dataset):

    def build_train_dataset(self, train_files):
        """Build dataset for training.

        Args:
            train_files: An iterable of tuple (src_file, tgt_file)

        Returns:
            A tf.data.Dataset object
        """
        dataset = self._build_dataset_from_files(train_files)
        dataset = self._shuffle(dataset)
        dataset = self._split_line(dataset)
        dataset = self._filter_dataset(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_sos_and_eos(dataset)
        batch_size = self.config.get('train_batch_size', 32)
        # padding with unk, may change to eos
        padding_value = tf.constant(self.src_tokenizer.unk_id, dtype=tf.dtypes.int64)
        dataset = self._padding_and_batching(dataset, batch_size=batch_size, padding_value=padding_value)
        return dataset

    def build_eval_dataset(self, eval_files):
        """Build dataset for evaluation.

        Args:
            eval_files: An iterable of tuple (src_file, tgt_file)

        Returns:
            A tf.data.Dataset object
        """
        dataset = self._build_dataset_from_files(eval_files)
        dataset = self._shuffle(dataset)
        dataset = self._split_line(dataset)
        dataset = self._filter_dataset(dataset)
        dataset = self._convert_tokens_to_ids(dataset)
        dataset = self._add_sos_and_eos(dataset)
        batch_size = self.config.get('eval_batch_size', 32)
        # padding with unk, may change to eos
        padding_value = tf.constant(self.src_tokenizer.unk_id, dtype=tf.dtypes.int64)
        dataset = self._padding_and_batching(dataset, batch_size=batch_size, padding_value=padding_value)
        return dataset

    def build_predict_dataset(self, predict_files):
        """Build dataset for prediction.

        Args:
            predict_files: An iterable of tuple (src_file)

        Returns:
            A tf.data.Dataset object
        """
        dataset = self._build_dataset_from_files_for_predict(predict_files)
        dataset = self._split_line_for_predict(dataset)
        dataset = self._convert_tokens_to_ids_for_predict(dataset)
        dataset = self._add_sos_and_eos_for_predict(dataset)
        batch_size = self.config.get('predict_batch_size', 32)
        # padding with unk, may change to eos
        padding_value = tf.constant(self.src_tokenizer.unk_id, dtype=tf.dtypes.int64)
        dataset = self._padding_and_batching_for_predict(dataset, batch_size=batch_size, padding_value=padding_value)
        return dataset
