import numpy as np
from . import iterutils, mathutils


def frand(size=None):
    return np.random.uniform(-1, 1, size=size)


class FeedForwardNetwork:
    def __init__(self, layer_sizes):
        self.ws = [frand(size=(x_size, y_size)) for x_size, y_size in iterutils.window(layer_sizes, 2)]
        self.bs = [frand(size=y_size) for _, y_size in iterutils.window(layer_sizes, 2)]

    def forward_prop(self, x0, intermediate_results):
        if "ys" not in intermediate_results:
            ys = []
            x = x0

            for w, b in zip(self.ws, self.bs):
                y = mathutils.sigmoid(np.dot(x, w) + b)
                ys.append(y)
                x = y

            intermediate_results["x0"] = x0
            intermediate_results["ys"] = ys

        return intermediate_results["ys"][-1]

    def back_prop(self, dy, intermediate_results):
        ys = intermediate_results["ys"]
        x0 = intermediate_results["x0"]

        dxs = []
        dx = dy
        for w, y in reversed(list(zip(self.ws, ys))):
            dx = np.dot(w, dx * mathutils.sigmoid_prime(y))
            dxs.append(dx)

        dxs.reverse()
        intermediate_results["dxs"] = dxs

        xs = [x0] + ys[:-1]
        dys = dxs[1:] + [dy]

        dws = []
        for x, w, y, dy in reversed(list(zip(xs, self.ws, ys, dys))):
            x2 = np.expand_dims(x, axis=1)

            dw = dy * mathutils.sigmoid_prime(y) * x2
            dws.append(dw)

        dws.reverse()
        intermediate_results["dws"] = dws

        dbs = []
        for w, y, dy in reversed(list(zip(self.ws, ys, dys))):
            db = dy * mathutils.sigmoid_prime(y)
            dbs.append(db)

        dbs.reverse()
        intermediate_results["dbs"] = dbs

        return dxs[0]

    def train(self, learning_rate, intermediate_results):
        dws = intermediate_results["dws"]
        dbs = intermediate_results["dbs"]

        for w, dw in zip(self.ws, dws):
            w -= dw * learning_rate

        for b, db in zip(self.bs, dbs):
            b -= db * learning_rate

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


