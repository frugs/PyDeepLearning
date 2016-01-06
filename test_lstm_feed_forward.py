import unittest
import numpy as np
from pydl import NoOutputLstm, FeedForwardNetwork


class TestLstmFeedForward(unittest.TestCase):
    def test_learn_word_vectors_from_char_vector_sequence(self):
        text = "please learn how to infer word vectors from sequences of character vectors"

        index_to_word = list(set(text.split()))
        index_to_char = list(set(text))

        word_to_index = {word: index for index, word in enumerate(index_to_word)}
        char_to_index = {word: index for index, word in enumerate(index_to_char)}

        def to_char_vector_sequence(word):
            sequence = []
            for char in word:
                vector = np.ones(len(char_to_index)) * -1
                vector[char_to_index[char]] = 1
                sequence.append(vector)
            sequence.append(np.zeros(len(char_to_index)))

            return np.asarray(sequence)

        def to_word_vector(word):
            vector = np.ones(len(word_to_index)) * -1
            vector[word_to_index[word]] = 1
            return vector

        training_data = [(to_char_vector_sequence(word), to_word_vector(word)) for word in text.split()]
        hidden_size = 100
        lstm = NoOutputLstm(len(index_to_char), hidden_size)
        ffn = FeedForwardNetwork([hidden_size, len(index_to_word)])

        learning_rate = 0.5

        for i in range(1000):
            for char_vectors, word_vector in training_data:
                hs, f_gs, i_gs, cs, lstm_output = lstm.forward_prop(char_vectors, np.zeros(hidden_size))
                lstm_out_grad, ffn_wb_grad = ffn.back_prop(np.expand_dims(lstm_output, axis=1), np.expand_dims(word_vector, axis=1))

                ffn_wb_delta = [(w * learning_rate, b * learning_rate) for w, b in ffn_wb_grad]
                ffn.apply_weight_and_bias_deltas(ffn_wb_delta)

                dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c = lstm.back_prop(char_vectors, hs, f_gs, i_gs, cs, lstm_out_grad.T[0])

                lstm.w_xf_g -= dw_xf_g * learning_rate
                lstm.w_hf_g -= dw_hf_g * learning_rate
                lstm.b_f_g -= db_f_g * learning_rate
                lstm.w_xi_g -= dw_xi_g * learning_rate
                lstm.w_hi_g -= dw_hi_g * learning_rate
                lstm.b_i_g -= db_i_g * learning_rate
                lstm.w_xc -= dw_xc * learning_rate
                lstm.w_hc -= dw_hc * learning_rate
                lstm.b_c -= db_c * learning_rate

            if i % 200 == 0:
                total_err = 0
                for char_vectors, word_vector in training_data:
                    h = lstm.activate(char_vectors, np.zeros(hidden_size))
                    output_vector = ffn.compute_outputs(np.expand_dims(h, axis=1))[-1].T[0]
                    total_err += np.sum(np.square(output_vector - word_vector))
                print(total_err/len(index_to_word))

        lstm_out = lstm.activate(to_char_vector_sequence("infer"), np.zeros(hidden_size))
        result = ffn.compute_outputs(np.expand_dims(h, axis=1))[-1].T[0]

        self.assertEquals("infer", index_to_word[np.argmax(result)])