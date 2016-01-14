from collections import namedtuple

import theano
import theano.tensor as T
import numpy as np
from . import iterutils
# import iterutils


def farray(arr):
    return np.array(arr).astype("float32")


def frand(size=None):
    return np.random.uniform(size=size).astype("float32")


t_x, t_b = T.fvectors("x", "b")
t_w = T.fmatrix("w")

t_activation = T.dot(t_x, t_w) + t_b
t_y = 1 / (1 + T.exp(-t_activation))

tf_activate = theano.function(inputs=[t_x, t_w, t_b], outputs=[t_y])

t_w2, t_x2 = T.fmatrices("w2", "x2")
t_dy, t_sigmoid = T.fvectors("dy", "sigmoid")
t_sigmoid_prime = t_sigmoid * (1 - t_sigmoid)
t_dw = t_dy * t_sigmoid_prime * t_x2
t_db = t_dy * t_sigmoid_prime
t_dx = T.dot(t_w2, t_dy * t_sigmoid_prime)

tf_dw = theano.function(inputs=[t_dy, t_sigmoid, t_x2], outputs=[t_dw])
tf_db = theano.function(inputs=[t_dy, t_sigmoid], outputs=[t_db])
tf_dx = theano.function(inputs=[t_dy, t_sigmoid, t_w2], outputs=[t_dx])


class FeedForwardNetwork:
    def __init__(self, layer_sizes):
        self.ws = [frand(size=(x_size, y_size)) for x_size, y_size in iterutils.window(layer_sizes, 2)]
        self.bs = [frand(size=y_size) for _, y_size in iterutils.window(layer_sizes, 2)]

    def y(self, x0, intermediate_results):
        if "ys" not in intermediate_results:
            ys = []
            x = x0

            for w, b in zip(self.ws, self.bs):
                y = tf_activate(x, w, b)[0]
                ys.append(y)
                x = y

            intermediate_results["ys"] = ys

        return intermediate_results["ys"][-1]

    def dy(self, x0, t, intermediate_results):
        if "dy" not in intermediate_results:
            y = self.y(x0, intermediate_results)
            intermediate_results["dy"] = y - t

        return intermediate_results["dy"]

    def dws(self, x0, dy, intermediate_results):
        if "dws" not in intermediate_results:
            dws = []
            ys = intermediate_results["ys"]
            xs = [x0] + ys[:-1]

            self.dx(x0, dy, intermediate_results)
            dxs = intermediate_results["dxs"]
            dys = dxs[1:] + [dy]

            for x, w, y, dy in reversed(list(zip(xs, self.ws, ys, dys))):
                x2 = np.expand_dims(x, axis=1)

                dw = tf_dw(dy, y, x2.repeat(len(y), axis=1))[0]
                dws.append(dw)

            dws.reverse()
            intermediate_results["dws"] = dws

        return intermediate_results["dws"]

    def dbs(self, x0, dy, intermediate_results):
        if "dbs" not in intermediate_results:
            dbs = []
            ys = intermediate_results["ys"]

            self.dx(x0, dy, intermediate_results)
            dxs = intermediate_results["dxs"]
            dys = dxs[1:] + [dy]

            for w, y, dy in reversed(list(zip(self.ws, ys, dys))):
                db = tf_db(dy, y)[0]
                dbs.append(db)

            dbs.reverse()
            intermediate_results["dbs"] = dbs

        return intermediate_results["dbs"]

    def dx(self, x0, dy, intermediate_results):
        if "dxs" not in intermediate_results:
            dxs = []
            ys = intermediate_results["ys"]

            for w, y in reversed(list(zip(self.ws, ys))):
                dy = tf_dx(dy, y, w)[0]
                dxs.append(dy)

            dxs.reverse()
            intermediate_results["dxs"] = dxs

        return intermediate_results["dxs"][0]

    def train(self, learning_rate, x0, dy, intermediate_results):
        dws = self.dws(x0, dy, intermediate_results)
        dbs = self.dbs(x0, dy, intermediate_results)
        intermediate_results.clear()

        for w, dw in zip(self.ws, dws):
            w -= dw * learning_rate

        for b, db in zip(self.bs, dbs):
            b -= db * learning_rate

    def train_from_results(self, learning_rate, intermediate_results):
        self.train(learning_rate, None, None, intermediate_results)

    def save(self, file_name):
        arrays = []
        for w, b in zip(self.ws, self.bs):
            arrays.append(w)
            arrays.append(b)
        np.savez_compressed(file_name, *arrays)

    def load(self, file_name):
        self.ws = []
        self.bs = []

        with np.load(file_name) as data:
            for i in range(0, len(data.items()), 2):
                self.ws.append(data["arr_%d" % i])
                self.bs.append(data["arr_%d" % (i + 1)])


