import unittest
import numpy as np
from pydl import NoOutputLstm, FeedForwardNetwork, mathutils


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
        # hidden_size = 100
        hidden_size = len(index_to_word)
        lstm = NoOutputLstm(len(index_to_char), hidden_size)
        ffn = FeedForwardNetwork([hidden_size, 50, 20, len(index_to_word)])

        h0 = np.random.uniform(-1, 1, size=hidden_size)

        learning_rate = 0.5

        for i in range(1000):
            for char_vectors, word_vector in training_data:
                hs, f_gs, i_gs, cs, lstm_output = lstm.forward_prop(char_vectors, h0)
                res = {}
                y = ffn.y(lstm_output, res)
                # dy = mathutils.mean_squared_error_prime(y, word_vector)
                dy = mathutils.mean_squared_error_prime(lstm_output, word_vector)
                dx = ffn.dx(lstm_output, dy, res)
                ffn.train(learning_rate, lstm_output, dy, res)

                # dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c = lstm.back_prop(char_vectors, hs, f_gs, i_gs, cs, dx)
                dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c = lstm.back_prop(char_vectors, hs, f_gs, i_gs, cs, dy)

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
                    h = lstm.activate(char_vectors, h0)
                    output_vector = ffn.y(h[-1], {})
                    total_err += mathutils.mean_squared_error(output_vector, word_vector)
                print(total_err/len(training_data))

        lstm_out = lstm.activate(to_char_vector_sequence("infer"), h0)
        result = ffn.y(lstm_out, {})

        self.assertEquals("infer", index_to_word[np.argmax(result)])

    def test_learn_word_vectors_from_char_vector_sequence_2(self):
        text = "please learn how to infer word vectors from sequences of character vectors" \
               "giving it more words to try and confuse it" \
               "how evil" \
               "much diabolical" \
               "many genius" \
               "the doge of venice gives his regards"

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

        hidden_size = 50

        training_data = [(to_char_vector_sequence(word), to_word_vector(word)) for word in text.split()]
        lstm = NoOutputLstm(len(index_to_char), hidden_size)
        ffn = FeedForwardNetwork([hidden_size, len(index_to_word)])

        h0 = np.random.uniform(-1, 1, size=hidden_size)

        learning_rate = 5

        for i in range(2000):
            for char_vectors, word_vector in training_data:
                hs, f_gs, i_gs, cs, h = lstm.forward_prop(char_vectors, h0)
                res = {}
                y = ffn.y(h, res)
                dy = mathutils.mean_squared_error(y, word_vector)
                dx = ffn.dx(h, dy, res)
                ffn.train(learning_rate, h, dy, res)
                dh = dx
                dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c = lstm.back_prop(char_vectors, hs, f_gs, i_gs, cs, dh)
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
                    h = lstm.activate(char_vectors, h0)
                    y = ffn.y(h, {})
                    total_err += mathutils.mean_squared_error(y, word_vector)
                print(total_err/len(training_data))

        h = lstm.activate(to_char_vector_sequence("infer"), h0)
        y = ffn.y(h, {})
        self.assertEquals("infer", index_to_word[np.argmax(y)])