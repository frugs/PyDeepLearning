import numpy as np
from .mathutils import sigmoid, sigmoid_prime, tanh_prime


def frand(size=None):
    return np.random.uniform(-0.2, 0.2, size=size)


class NoOutputLstm:
    def __init__(self, input_size: int, hidden_size: int):
        self.w_xf_g = frand(size=(input_size, hidden_size))
        self.w_hf_g = frand(size=(hidden_size, hidden_size))
        self.b_f_g = frand(size=hidden_size)
        self.w_xi_g = frand(size=(input_size, hidden_size))
        self.w_hi_g = frand(size=(hidden_size, hidden_size))
        self.b_i_g = frand(size=hidden_size)
        self.w_xc = frand(size=(input_size, hidden_size))
        self.w_hc = frand(size=(hidden_size, hidden_size))
        self.b_c = frand(size=hidden_size)

    def clone(self):
        clone = NoOutputLstm(0, 0)
        clone.w_xf_g = np.copy(self.w_xf_g)
        clone.w_hf_g = np.copy(self.w_hf_g)
        clone.b_f_g = np.copy(self.b_f_g)
        clone.w_xi_g = np.copy(self.w_xi_g)
        clone.w_hi_g = np.copy(self.w_hi_g)
        clone.b_i_g = np.copy(self.b_i_g)
        clone.w_xc = np.copy(self.w_xc)
        clone.w_hc = np.copy(self.w_hc)
        clone.b_c = np.copy(self.b_c)
        return clone

    def _step(self, x, h_prev, intermediate_results):
        f_g = sigmoid(np.dot(x, self.w_xf_g) + np.dot(h_prev, self.w_hf_g) + self.b_f_g)
        i_g = sigmoid(np.dot(x, self.w_xi_g) + np.dot(h_prev, self.w_hi_g) + self.b_i_g)
        c = np.tanh(np.dot(x, self.w_xc) + np.dot(h_prev, self.w_hc) + self.b_c)
        h = h_prev * f_g + i_g * c

        intermediate_results["f_gs"].append(f_g)
        intermediate_results["i_gs"].append(i_g)
        intermediate_results["cs"].append(c)
        intermediate_results["hs"].append(h)
        return h

    def forward_prop(self, xs, h0, intermediate_results):
        if "hs" not in intermediate_results:
            intermediate_results["h0"] = h0
            intermediate_results["xs"] = xs
            intermediate_results["hs"] = []
            intermediate_results["f_gs"] = []
            intermediate_results["i_gs"] = []
            intermediate_results["cs"] = []

            h = h0

            for x in xs:
                h = self._step(x, h, intermediate_results)

        return intermediate_results["hs"][-1]

    def _back_step(self, x, h_prev, f_g, i_g, c, dh, intermediate_results):
        df_g = dh * h_prev
        dact_f_g = df_g * sigmoid_prime(f_g)
        dw_xf_g = dact_f_g * np.expand_dims(x, axis=1)
        dw_hf_g = dact_f_g * np.expand_dims(h_prev, axis=1)
        db_f_g = dact_f_g

        dh_prev = np.dot(self.w_hf_g, dact_f_g)

        di = dh * c
        dact_i_g = di * sigmoid_prime(i_g)
        dw_xi_g = dact_i_g * np.expand_dims(x, axis=1)
        dw_hi_g = dact_i_g * np.expand_dims(h_prev, axis=1)
        db_i_g = dact_i_g

        dh_prev += np.dot(self.w_hi_g, dact_i_g)

        dc = dh * i_g
        dact_c = dc * tanh_prime(c)
        dw_xc = dact_c * np.expand_dims(x, axis=1)
        dw_hc = dact_c * np.expand_dims(h_prev, axis=1)
        db_c = dact_c

        dh_prev += np.dot(self.w_hc, dact_c)

        dh_prev += dh * f_g

        intermediate_results["dw_xf_g"] += dw_xf_g
        intermediate_results["dw_hf_g"] += dw_hf_g
        intermediate_results["db_f_g"] += db_f_g
        intermediate_results["dw_xi_g"] += dw_xi_g
        intermediate_results["dw_hi_g"] += dw_hi_g
        intermediate_results["db_i_g"] += db_i_g
        intermediate_results["dw_xc"] += dw_xc
        intermediate_results["dw_hc"] += dw_hc
        intermediate_results["db_c"] += db_c

        return dh_prev

    def back_prop(self, dh_last, intermediate_results):
        h0 = intermediate_results["h0"]
        xs = intermediate_results["xs"]
        hs = intermediate_results["hs"]
        f_gs = intermediate_results["f_gs"]
        i_gs = intermediate_results["i_gs"]
        cs = intermediate_results["cs"]

        intermediate_results["dw_xf_g"] = np.zeros(self.w_xf_g.shape)
        intermediate_results["dw_hf_g"] = np.zeros(self.w_hf_g.shape)
        intermediate_results["db_f_g"] = np.zeros(self.b_f_g.shape)
        intermediate_results["dw_xi_g"] = np.zeros(self.w_xi_g.shape)
        intermediate_results["dw_hi_g"] = np.zeros(self.w_hi_g.shape)
        intermediate_results["db_i_g"] = np.zeros(self.b_i_g.shape)
        intermediate_results["dw_xc"] = np.zeros(self.w_xc.shape)
        intermediate_results["dw_hc"] = np.zeros(self.w_hc.shape)
        intermediate_results["db_c"] = np.zeros(self.b_c.shape)

        dh = dh_last
        for x, h_prev, f_g, i_g, c in reversed(list(zip(xs, [h0] + hs[:-1], f_gs, i_gs, cs))):
            dh_prev = self._back_step(x, h_prev, f_g, i_g, c, dh, intermediate_results)
            dh = dh_prev

        return dh

    def activate(self, xs, h0):
        return self.forward_prop(xs, h0, {})

    def train_from_results(self, learning_rate, intermediate_results):
        self.w_xf_g -= intermediate_results["dw_xf_g"] * learning_rate
        self.w_hf_g -= intermediate_results["dw_hf_g"] * learning_rate
        self.b_f_g -= intermediate_results["db_f_g"] * learning_rate
        self.w_xi_g -= intermediate_results["dw_xi_g"] * learning_rate
        self.w_hi_g -= intermediate_results["dw_hi_g"] * learning_rate
        self.b_i_g -= intermediate_results["db_i_g"] * learning_rate
        self.w_xc -= intermediate_results["dw_xc"] * learning_rate
        self.w_hc -= intermediate_results["dw_hc"] * learning_rate
        self.b_c -= intermediate_results["db_c"] * learning_rate