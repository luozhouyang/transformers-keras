import unittest

from .transformer_tokenizer import TransformerDefaultTokenizer
from .transformer_tokenizer import TransformerJiebaTokenizer


class TransformerTokenizerTest(unittest.TestCase):

    def testTransformerDefaultTokenizer(self):
        tokenizer = TransformerDefaultTokenizer(
            'testdata/vocab_src.txt',
            unk_token='<unk>', sos_token='<s>', eos_token='</s>', pad_token='<pad>')
        # print(tokenizer.token2id)
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(1, tokenizer.unk_id)
        self.assertEqual(2, tokenizer.sos_id)
        self.assertEqual(3, tokenizer.eos_id)
        self.assertEqual('<pad>', tokenizer.pad_token)
        self.assertEqual('<unk>', tokenizer.unk_token)
        self.assertEqual('<s>', tokenizer.sos_token)
        self.assertEqual('</s>', tokenizer.eos_token)

        seq = '我 住在 上海'
        self.assertEqual(['我', '住', '在', '上', '海'], tokenizer.tokenize(seq))
        self.assertEqual([4, 5, 6, 7, 8], tokenizer.encode(seq))

    def testTransformerJiebaTokenizer(self):
        tokenizer = TransformerJiebaTokenizer(
            'testdata/vocab_src.txt',
            unk_token='<unk>', sos_token='<s>', eos_token='</s>', pad_token='<pad>')
        print(tokenizer.token2id)
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(1, tokenizer.unk_id)
        self.assertEqual(2, tokenizer.sos_id)
        self.assertEqual(3, tokenizer.eos_id)
        self.assertEqual('<pad>', tokenizer.pad_token)
        self.assertEqual('<unk>', tokenizer.unk_token)
        self.assertEqual('<s>', tokenizer.sos_token)
        self.assertEqual('</s>', tokenizer.eos_token)

        seq = '我 住在 上海'
        self.assertEqual(['我', '住', '在', '上海'], tokenizer.tokenize(seq))
        self.assertEqual([4, 5, 6, 1], tokenizer.encode(seq))


if __name__ == "__main__":
    unittest.main()
