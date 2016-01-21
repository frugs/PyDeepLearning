import unittest
import time
import numpy as np
from pydl import NoOutputLstm, mathutils


def err(y):
    return np.sum(np.square(y)) / len(y)


def derr(y):
    return (2 / len(y)) * y


class TestNoOutputLstm(unittest.TestCase):
    def test_step_gradients(self):
        t = 1e-4

        x = np.random.randn(4)
        h_prev = np.random.randn(5)

        n = NoOutputLstm(len(x), len(h_prev))

        h_next, f_g, i_g, c = n._step(x, h_prev)
        dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c, dh_prev = n._back_step(x, h_prev, f_g, i_g, c, derr(h_next))

        def grad_check(attribute, numerical_gradient):
            for i in np.ndindex(numerical_gradient.shape):
                plus_n = n.clone()
                getattr(plus_n, attribute)[i] += t

                neg_n = n.clone()
                getattr(neg_n, attribute)[i] -= t

                plus_h_next, _, _, _ = plus_n._step(x, h_prev)
                neg_h_next, _, _, _ = neg_n._step(x, h_prev)
                exp_grad = np.sum((err(plus_h_next) - err(neg_h_next)) / (2 * t))

                self.assertTrue(abs(exp_grad - numerical_gradient[i]) < 0.01,
                                "{}: {} not within threshold of {}".format(attribute, numerical_gradient[i], exp_grad))
        checks = {
            "w_xf_g": dw_xf_g,
            "w_hf_g": dw_hf_g,
            "b_f_g": db_f_g,
            "w_xi_g": dw_xi_g,
            "w_hi_g": dw_hi_g,
            "b_i_g": db_i_g,
            "w_xc": dw_xc,
            "w_hc": dw_hc,
            "b_c": db_c
        }

        for attr, numerical_grad in checks.items():
            grad_check(attr, numerical_grad)

        for i in np.ndindex(dh_prev.shape):
            hprev_plus = np.copy(h_prev)
            hprev_plus[i] += t

            hprev_minus = np.copy(h_prev)
            hprev_minus[i] -= t

            plus_h1, _, _, _ = n._step(x, hprev_plus)
            neg_h1, _, _, _ = n._step(x, hprev_minus)
            exp_dh_prev = ((err(plus_h1) - err(neg_h1)) / (2 * t))

            self.assertTrue(abs(exp_dh_prev - dh_prev[i]) < 0.01,
                            "dh_prev: {} not within threshold of {}".format(dh_prev[i], exp_dh_prev))

    def test_prop_gradients(self):
        t = 1e-4

        x_size = 4
        xs = np.random.randn(10, x_size)
        h0 = np.random.randn(5)

        n = NoOutputLstm(x_size, len(h0))
        hs, f_gs, i_gs, cs, _ = n.forward_prop(xs, h0)
        dw_xf_g, dw_hf_g, db_f_g, dw_xi_g, dw_hi_g, db_i_g, dw_xc, dw_hc, db_c = n.back_prop(xs, hs, f_gs, i_gs, cs, np.ones(h0.shape))

        def grad_check(attribute, numerical_gradient):
            for i in np.ndindex(numerical_gradient.shape):
                plus_n = n.clone()
                getattr(plus_n, attribute)[i] += t

                neg_n = n.clone()
                getattr(neg_n, attribute)[i] -= t

                _, _, _, _, plus_h_next = plus_n.forward_prop(xs, h0)
                _, _, _, _, neg_h_next = neg_n.forward_prop(xs, h0)
                exp_grad = np.sum((plus_h_next - neg_h_next) / (2 * t))
                num_grad = numerical_gradient[i]

                self.assertTrue(abs(exp_grad - num_grad) < 0.01,
                                "{}: {} not within threshold of {}".format(attribute, numerical_gradient[i], exp_grad))
        checks = {
            "w_xf_g": dw_xf_g,
            "w_hf_g": dw_hf_g,
            "b_f_g": db_f_g,
            "w_xi_g": dw_xi_g,
            "w_hi_g": dw_hi_g,
            "b_i_g": db_i_g,
            "w_xc": dw_xc,
            "w_hc": dw_hc,
            "b_c": db_c
        }

        for attr, numerical_grad in checks.items():
            grad_check(attr, numerical_grad)

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
        n = NoOutputLstm(len(index_to_char), len(index_to_word))

        for i in range(1000):
            for char_vectors, word_vector in training_data:
                n.train(char_vectors, np.zeros(len(index_to_word)), word_vector, 0.1)

            if i % 200 == 0:
                total_err = 0
                for char_vectors, word_vector in training_data:
                    h = n.activate(char_vectors, np.zeros(len(index_to_word)))
                    total_err += mathutils.mean_squared_error(h, word_vector)
                print(total_err/len(training_data))

        result = n.activate(to_char_vector_sequence("infer"), np.zeros(len(index_to_word)))
        self.assertEquals("infer", index_to_word[np.argmax(result)])

    def test_training_performance(self):
        n = NoOutputLstm(100, 80)
        training_data = []
        for _ in range(30):
            xs = np.asarray([np.random.uniform(size=100),
                             np.random.uniform(size=100),
                             np.random.uniform(size=100),
                             np.random.uniform(size=100),
                             np.random.uniform(size=100),
                             np.random.uniform(size=100),
                             np.random.uniform(size=100),
                             np.random.uniform(size=100)])
            h0 = np.random.uniform(size=80)
            t = np.random.uniform(size=80)
            training_data.append((xs, h0, t))

        epochs = 300
        start = time.time()
        n.train_from_data(training_data, 0.1, epochs=epochs)
        end = time.time()
        time_taken = end - start

        print(str(epochs) + " training epochs took " + str(time_taken) + " seconds")

if __name__ == '__main__':
    unittest.main()