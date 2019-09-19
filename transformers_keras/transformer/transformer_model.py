import tensorflow as tf

from transformers_keras.transformer.decoder import Decoder
from transformers_keras.transformer.encoder import Encoder


class Transformer(tf.keras.Model):

    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=None, mask=None):
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = mask
        enc_output, enc_attn_weights = self.encoder(inp, training,
                                                    enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, dec_attn_weights = self.decoder(
            (tar, enc_output), training, (look_ahead_mask, dec_padding_mask))

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, enc_attn_weights, dec_attn_weights
