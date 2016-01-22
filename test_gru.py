import unittest
import time
import numpy as np
from pydl import Gru, mathutils


def frand(size=None):
    return np.random.uniform(-1, 1, size=size)


def err(y):
    return np.sum(0.5 * np.square(y))


def derr(y):
    return y


def ce_err_prime(y, t):
    return y - t


def gru_arrays(gru: Gru):
    return [
        gru.w_rx,
        gru.w_rh,
        gru.b_r,
        gru.w_zx,
        gru.w_zh,
        gru.b_z,
        gru.w_hx,
        gru.w_hh,
        gru.b_h
    ]


def gru_array_names(gru: Gru):
    return [
        "w_rx",
        "w_rh",
        "b_r",
        "w_zx",
        "w_zh",
        "b_z",
        "w_hx",
        "w_hh",
        "b_h"
    ]


def gru_results_arrays(gru_results: dict):
    return [
        gru_results["dw_rx"],
        gru_results["dw_rh"],
        gru_results["db_r"],
        gru_results["dw_zx"],
        gru_results["dw_zh"],
        gru_results["db_z"],
        gru_results["dw_hx"],
        gru_results["dw_hh"],
        gru_results["db_h"]
    ]


def scalar_indices(gru: Gru):
    for i, array in enumerate(gru_arrays(gru)):
        for j in np.ndindex(array.shape):
            yield i, j


def clone(gru: Gru):
    gru_clone = Gru(0, 0)
    gru_clone.w_rx = np.copy(gru.w_rx)
    gru_clone.w_rh = np.copy(gru.w_rh)
    gru_clone.b_r = np.copy(gru.b_r)
    gru_clone.w_zx = np.copy(gru.w_zx)
    gru_clone.w_zh = np.copy(gru.w_zh)
    gru_clone.b_z = np.copy(gru.b_z)
    gru_clone.w_hx = np.copy(gru.w_hx)
    gru_clone.w_hh = np.copy(gru.w_hh)
    gru_clone.b_h = np.copy(gru.b_h)
    return gru_clone


class TestGru(unittest.TestCase):
    def test_single_step_gradient(self):
        input_size = 5
        hidden_size = 6
        n = Gru(input_size, hidden_size)

        xs = [frand(size=input_size)]
        h0 = frand(hidden_size)

        intermediate_results = {}
        hs = n.forward_prop(xs, h0, intermediate_results)
        dh0 = n.back_prop([derr(hs[-1])], intermediate_results)

        delta = 1e-4

        for index in scalar_indices(n):
            array_name = gru_array_names(n)[index[0]]

            slightly_less = clone(n)
            gru_arrays(slightly_less)[index[0]][index[1]] -= delta
            err_slightly_less = err(slightly_less.forward_prop(xs, h0, {})[-1])

            slightly_more = clone(n)
            gru_arrays(slightly_more)[index[0]][index[1]] += delta
            err_slightly_more = err(slightly_more.forward_prop(xs, h0, {})[-1])

            expected_grad = (err_slightly_more - err_slightly_less) / (2 * delta)
            numerical_grad = gru_results_arrays(intermediate_results)[index[0]][index[1]]

            self.assertTrue(abs(expected_grad - numerical_grad) < 0.01,
                            "{}: {} not within threshold of {}".format(array_name, numerical_grad, expected_grad))

        for index in np.ndindex(h0.shape):
            slightly_less_h0 = np.copy(h0)
            slightly_less_h0[index] -= delta
            err_slightly_less_h0 = err(n.forward_prop(xs, slightly_less_h0, {})[-1])

            slightly_more_h0 = np.copy(h0)
            slightly_more_h0[index] += delta
            err_slightly_more_h0 = err(n.forward_prop(xs, slightly_more_h0, {})[-1])

            expected_grad = (err_slightly_more_h0 - err_slightly_less_h0) / (2 * delta)
            numerical_grad = dh0[index]

            self.assertTrue(abs(expected_grad - numerical_grad) < 0.01,
                            "h0: {} not within threshold of {}".format(numerical_grad, expected_grad))

    def test_multi_step_gradient(self):
        input_size = 5
        hidden_size = 6
        n = Gru(input_size, hidden_size)

        xs = [frand(size=input_size) for _ in range(10)]
        h0 = frand(hidden_size)

        intermediate_results = {}
        hs = n.forward_prop(xs, h0, intermediate_results)
        n.back_prop([derr(h) for h in hs], intermediate_results)

        for index in scalar_indices(n):
            array_name = gru_array_names(n)[index[0]]
            delta = 1e-4

            slightly_less = clone(n)
            gru_arrays(slightly_less)[index[0]][index[1]] -= delta
            slightly_less_hs = slightly_less.forward_prop(xs, h0, {})
            err_slightly_less = sum([err(h) for h in slightly_less_hs])

            slightly_more = clone(n)
            gru_arrays(slightly_more)[index[0]][index[1]] += delta
            slightly_more_hs = slightly_more.forward_prop(xs, h0, {})
            err_slightly_more = sum([err(h) for h in slightly_more_hs])

            expected_grad = (err_slightly_more - err_slightly_less) / (2 * delta)
            numerical_grad = gru_results_arrays(intermediate_results)[index[0]][index[1]]

            self.assertTrue(abs(expected_grad - numerical_grad) < 0.01,
                            "{}: {} not within threshold of {}".format(array_name, numerical_grad, expected_grad))

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

            return sequence

        def to_word_vector(word):
            vector = np.ones(len(word_to_index)) * -1
            vector[word_to_index[word]] = 1
            return vector

        training_data = [(to_char_vector_sequence(word), to_word_vector(word)) for word in text.split()]
        n = Gru(len(index_to_char), len(index_to_word))

        for i in range(1000):
            for char_vectors, word_vector in training_data:
                intermediate_results = {}
                hs = n.forward_prop(char_vectors, np.zeros(len(index_to_word)), intermediate_results)
                dhs = [np.zeros(shape=word_vector.shape) for _ in range(len(hs))]
                dhs[-1] = ce_err_prime(hs[-1], word_vector)
                n.back_prop(dhs, intermediate_results)
                n.train(0.1, intermediate_results)

            if i % 200 == 0:
                total_err = 0
                for char_vectors, word_vector in training_data:
                    hs = n.forward_prop(char_vectors, np.zeros(len(index_to_word)), {})
                    total_err += mathutils.mse(hs[-1], word_vector)
                print(total_err/len(training_data))

        result = n.forward_prop(to_char_vector_sequence("infer"), np.zeros(len(index_to_word)), {})[-1]
        self.assertEquals("infer", index_to_word[np.argmax(result)])