import os
import random
import tempfile
import unittest
import numpy as np
import numpy.testing as npt
from pydl.feedforwardnetwork import FeedForwardNetwork
from pydl import mathutils


def clone(n):
    n2 = FeedForwardNetwork([9, 9])
    n2.ws = [np.copy(w) for w in n.ws]
    n2.bs = [np.copy(b) for b in n.bs]
    return n2


def err(y):
    return mathutils.mean_squared_error(y, np.zeros(y.shape))


class TestFeedForwardNetwork(unittest.TestCase):
    def test_grad_ws(self):
        n = FeedForwardNetwork([5, 4, 3, 2])
        x0 = np.random.uniform(size=5).astype("float32")

        res = {}
        y = n.y(x0, res)
        t = np.zeros(2).astype("float32")
        dy = mathutils.mean_squared_error_prime(y, t)

        dws = n.dws(x0, dy, res)

        delta = 1e-4

        exp_dws = []
        for i in range(len(n.ws)):
            w = n.ws[i]
            exp_dw = np.zeros(w.shape)
            for index in np.ndindex(w.shape):
                n1 = clone(n)
                n2 = clone(n)

                n1.ws[i][index] -= delta
                n2.ws[i][index] += delta

                exp_grad = (err(n2.y(x0, {})) - err(n1.y(x0, {}))) / (2 * delta)
                exp_dw[index] = exp_grad

            exp_dws.append(exp_dw)

        for dw, exp_dw in zip(dws, exp_dws):
            npt.assert_array_almost_equal(dw, exp_dw, decimal=3)

    def test_grad_bs(self):
        n = FeedForwardNetwork([4, 7, 2, 3])
        x0 = np.random.uniform(size=4).astype("float32")

        res = {}
        y = n.y(x0, res)
        t = np.zeros(3).astype("float32")
        dy = mathutils.mean_squared_error_prime(y, t)

        dbs = n.dbs(x0, dy, res)

        delta = 1e-4

        exp_dbs = []
        for i in range(len(n.bs)):
            b = n.bs[i]
            exp_db = np.zeros(b.shape)
            for index in np.ndindex(b.shape):
                n1 = clone(n)
                n2 = clone(n)

                n1.bs[i][index] -= delta
                n2.bs[i][index] += delta

                exp_grad = (err(n2.y(x0, {})) - err(n1.y(x0, {}))) / (2 * delta)
                exp_db[index] = exp_grad

            exp_dbs.append(exp_db)

        for dw, exp_db in zip(dbs, exp_dbs):
            npt.assert_array_almost_equal(dw, exp_db, decimal=3)

    def test_grad_x(self):
        n = FeedForwardNetwork([3, 4, 4, 2])
        x0 = np.random.uniform(size=3).astype("float32")

        res = {}
        y = n.y(x0, res)
        t = np.zeros(2).astype("float32")
        dy = mathutils.mean_squared_error_prime(y, t)
        dx = n.dx(x0, dy, res)

        delta = 1e-4

        exp_dx = np.zeros(x0.shape)
        for index in np.ndindex(x0.shape):
            x0_a = np.copy(x0)
            x0_b = np.copy(x0)

            x0_a[index] -= delta
            x0_b[index] += delta

            exp_grad = (err(n.y(x0_b, {})) - err(n.y(x0_a, {}))) / (2 * delta)
            exp_dx[index] = exp_grad

        npt.assert_array_almost_equal(dx, exp_dx, decimal=3)

    def test_save_and_load(self):
        n = FeedForwardNetwork([2, 3, 4])
        n.ws = [np.array([[1, 1],
                          [0, -1],
                          [5, -9]]),
                np.array([[2, 7, -4],
                          [0, -1, 1],
                          [6, 20, -10],
                          [3, 3, 3, 3]])]

        n.bs = [np.array([[0], [0], [1]]),
                np.array([[9], [1], [-1], [50]])]

        temp_file = tempfile.mkstemp(suffix=".npz")[1]
        n.save(temp_file)

        n2 = FeedForwardNetwork([])
        n2.load(temp_file)

        for w, b, w2, b2 in zip(n.ws, n.bs, n2.ws, n2.bs):
            npt.assert_equal(w, w2)
            npt.assert_equal(b, b2)

        os.remove(temp_file)

    def test_iris_data_set(self):
        def create_data_entry(line):
            split = line.strip().split(",")
            data_input = np.array([float(str) for str in split[:-1]]).astype("float32")

            classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
            data_target = np.array([float(split[-1] == class_) for class_ in classes]).astype("float32")

            return data_input, data_target

        iris_data_file = open("iris.data")
        data_set = [create_data_entry(line) for line in iris_data_file.readlines() if line.strip()]
        iris_data_file.close()
        random.shuffle(data_set)

        training_set = data_set[:-30]
        test_set = data_set[-30:]

        n = FeedForwardNetwork([4, 15, 3])
        learning_rate = 0.1

        for _ in range(10000):
            training_input, training_target = training_set[random.randrange(0, len(training_set))]
            res = {}
            y = n.y(training_input, res)
            dy = mathutils.mean_squared_error_prime(y, training_target)
            n.train(learning_rate, training_input, dy, res)

        errors = [mathutils.mean_squared_error(n.y(test_input, {}), test_target) for test_input, test_target in test_set]
        mean_squared_error = np.mean(np.square(errors))
        print(mean_squared_error)
        npt.assert_array_less(mean_squared_error, 0.05)
