import numpy as np
from . import mathutils


def frand(size=None):
    return np.random.uniform(-0.2, 0.2, size=size)


class Gru:
    def __init__(self, input_size, hidden_size):
        self.w_rx = frand(size=(input_size, hidden_size))
        self.w_rh = frand(size=(hidden_size, hidden_size))
        self.b_r = frand(size=hidden_size)

        self.w_zx = frand(size=(input_size, hidden_size))
        self.w_zh = frand(size=(hidden_size, hidden_size))
        self.b_z = frand(size=hidden_size)

        self.w_hx = frand(size=(input_size, hidden_size))
        self.w_hh = frand(size=(hidden_size, hidden_size))
        self.b_h = frand(size=hidden_size)

    def forward_prop(self, xs, h0: np.ndarray, intermediate_results: dict):
        intermediate_results["h0"] = h0
        intermediate_results["xs"] = xs
        intermediate_results["rs"] = []
        intermediate_results["zs"] = []
        intermediate_results["h_tildes"] = []
        intermediate_results["hs"] = []

        h = h0
        for x in xs:
            r = mathutils.sigmoid(np.dot(x, self.w_rx) + np.dot(h, self.w_rh) + self.b_r)
            z = mathutils.sigmoid(np.dot(x, self.w_zx) + np.dot(h, self.w_zh) + self.b_z)
            h_tilde = np.tanh(np.dot(x, self.w_hx) + np.dot(r * h, self.w_hh) + self.b_h)
            h = z * h + (1 - z) * h_tilde

            intermediate_results["rs"].append(r)
            intermediate_results["zs"].append(z)
            intermediate_results["h_tildes"].append(h_tilde)
            intermediate_results["hs"].append(h)

        return intermediate_results["hs"]

    def back_prop(self, dhs, intermediate_results):
        xs = intermediate_results["xs"]
        h0 = intermediate_results["h0"]
        hs = intermediate_results["hs"]
        h_prevs = [h0] + hs[:-1]
        h_tildes = intermediate_results["h_tildes"]
        zs = intermediate_results["zs"]
        rs = intermediate_results["rs"]

        intermediate_results["dw_rx"] = np.zeros(self.w_rx.shape)
        intermediate_results["dw_rh"] = np.zeros(self.w_rh.shape)
        intermediate_results["db_r"] = np.zeros(self.b_r.shape)

        intermediate_results["dw_zx"] = np.zeros(self.w_zx.shape)
        intermediate_results["dw_zh"] = np.zeros(self.w_zh.shape)
        intermediate_results["db_z"] = np.zeros(self.b_z.shape)

        intermediate_results["dw_hx"] = np.zeros(self.w_hx.shape)
        intermediate_results["dw_hh"] = np.zeros(self.w_hh.shape)
        intermediate_results["db_h"] = np.zeros(self.b_h.shape)

        dh_propagated = np.zeros(h0.shape)
        for x, h_prev, h_tilde, z, r, dh in reversed(list(zip(xs, h_prevs, h_tildes, zs, rs, dhs))):
            dh += dh_propagated

            dh_prev = dh * z

            dz = dh * (h_prev - h_tilde)

            db_z = dz * mathutils.sigmoid_prime(z)
            dw_zx = db_z * np.expand_dims(x, axis=1)
            dw_zh = db_z * np.expand_dims(h_prev, axis=1)

            dh_prev += np.dot(self.w_zh, db_z)

            dh_tilde = dh * (1 - z)
            db_h = dh_tilde * mathutils.tanh_prime(h_tilde)
            dw_hx = db_h * np.expand_dims(x, axis=1)
            dw_hh = db_h * np.expand_dims(r * h_prev, axis=1)

            dh_prev += np.dot(self.w_hh, db_h) * r

            dr = np.dot(self.w_hh, db_h) * h_prev
            db_r = dr * mathutils.sigmoid_prime(r)
            dw_rx = db_r * np.expand_dims(x, axis=1)
            dw_rh = db_r * np.expand_dims(h_prev, axis=1)

            dh_prev += np.dot(self.w_rh, db_r)

            intermediate_results["dw_rx"] += dw_rx
            intermediate_results["dw_rh"] += dw_rh
            intermediate_results["db_r"] += db_r

            intermediate_results["dw_zx"] += dw_zx
            intermediate_results["dw_zh"] += dw_zh
            intermediate_results["db_z"] += db_z

            intermediate_results["dw_hx"] += dw_hx
            intermediate_results["dw_hh"] += dw_hh
            intermediate_results["db_h"] += db_h

            dh_propagated = dh_prev

        return dh_propagated

    def train(self, learning_rate, intermediate_results):
        self.w_rx -= learning_rate * intermediate_results["dw_rx"]
        self.w_rh -= learning_rate * intermediate_results["dw_rh"]
        self.b_r -= learning_rate * intermediate_results["db_r"]

        self.w_zx -= learning_rate * intermediate_results["dw_zx"]
        self.w_zh -= learning_rate * intermediate_results["dw_zh"]
        self.b_z -= learning_rate * intermediate_results["db_z"]

        self.w_hx -= learning_rate * intermediate_results["dw_hx"]
        self.w_hh -= learning_rate * intermediate_results["dw_hh"]
        self.b_h -= learning_rate * intermediate_results["db_h"]