import unittest

from .bert_tokenizer import BertDefaultTokenizer


class BertTokenizerTest(unittest.TestCase):

    def testBertDefaultTokenizer(self):
        tokenizer = BertDefaultTokenizer(
            'testdata/vocab_src.txt',
            unk_token='<unk>', sos_token='<s>', eos_token='</s>', pad_token='<pad>')
        self.assertEqual(0, tokenizer.pad_id)
        self.assertEqual(1, tokenizer.unk_id)
        self.assertEqual(2, tokenizer.sos_id)
        self.assertEqual(3, tokenizer.eos_id)
        self.assertEqual(4, tokenizer.cls_id)
        self.assertEqual(5, tokenizer.sep_id)
        self.assertEqual(6, tokenizer.mask_id)
        self.assertEqual('<pad>', tokenizer.pad_token)
        self.assertEqual('<unk>', tokenizer.unk_token)
        self.assertEqual('<s>', tokenizer.sos_token)
        self.assertEqual('</s>', tokenizer.eos_token)
        self.assertEqual('[CLS]', tokenizer.cls_token)
        self.assertEqual('[SEP]', tokenizer.sep_token)
        self.assertEqual('[MASK]', tokenizer.mask_token)

        seq = '我 住在 上海'
        self.assertEqual(['我', '住', '在', '上', '海'], tokenizer.tokenize(seq))
        self.assertEqual([7, 8, 9, 10, 11], tokenizer.encode(seq))
        self.assertEqual(12, tokenizer.vocab_size)


if __name__ == "__main__":
    unittest.main()
