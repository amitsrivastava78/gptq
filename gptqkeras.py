import math
import time
import tensorflow as tf
import keras
import numpy as np

ops = tf  # Keras 3.0 ops API

DEBUG = False

# Disable TensorFlow optimizations for consistency
tf.config.optimizer.set_jit(False)

# Helper to robustly cast to int
def to_python_int(x):
    if hasattr(x, 'numpy'):
        return int(x.numpy())
    return int(x)

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        # Get weight tensor (equivalent to layer.weight.data.clone())
        W = tf.convert_to_tensor(layer.weights[0].numpy())
        if isinstance(self.layer, keras.layers.Conv2D):
            W = tf.reshape(W, [W.shape[0], -1])
        # Note: No Conv1D equivalent in Keras, so we skip that check
        self.rows = int(W.shape[0])
        self.columns = int(W.shape[1])
        self.H = tf.zeros((self.rows, self.rows), dtype=tf.float32)
        self.nsamples = 0
        self.quantizer = None

    # def add_batch(self, inp, out):
    #     if DEBUG:
    #         self.inp1 = inp
    #         self.out1 = out
    #     if len(inp.shape) == 2:
    #         inp = tf.expand_dims(inp, 0)
    #     tmp = inp.shape[0]
    #     if isinstance(self.layer, keras.layers.Dense):
    #         if len(inp.shape) == 3:
    #             inp = tf.reshape(inp, [-1, inp.shape[-1]])
    #         inp = tf.transpose(inp)
    #     print("Shape before matmul:", inp.shape)
    #     if isinstance(self.layer, keras.layers.Conv2D):
    #         # Keras doesn't have Unfold, so we'll skip this for now
    #         # This would need a custom implementation for Conv2D
    #         pass
    #     self.H = self.H * (self.nsamples / (self.nsamples + tmp))
    #     self.nsamples += tmp
    #     inp = math.sqrt(2 / self.nsamples) * tf.cast(inp, tf.float32)
    #     self.H = self.H + tf.matmul(inp, tf.transpose(inp))

    def add_batch(self, inp, out):
    # --- Corrected Logic ---
        print("Inside GPTQ add_batch")
        # 1. Reshape 3D inputs to 2D. This leaves 2D inputs unchanged.
        if len(inp.shape) == 3:
            inp = tf.reshape(inp, [-1, inp.shape[-1]])  # [batch*seq, features]
        inp = tf.transpose(inp)  # [features, batch*seq]
        print("self.H shape:", self.H.shape)
        print("inp shape:", inp.shape)
        print("matmul shape:", tf.matmul(inp, tf.transpose(inp)).shape)
        self.H = self.H + tf.matmul(inp, tf.transpose(inp))  # [features, features]

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False):
        W = tf.convert_to_tensor(self.layer.weights[0].numpy(), dtype=tf.float32)
        if isinstance(self.layer, keras.layers.Conv2D):
            W = tf.reshape(W, [W.shape[0], -1])
        # Note: No Conv1D equivalent in Keras

        tick = time.time()

        if self.quantizer is not None and self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        
        # Check if we have any calibration data
        if self.nsamples == 0:
            print("WARNING: No calibration data collected. Using identity Hessian.")
            H = tf.eye(self.columns, dtype=tf.float32)
        else:
            dead = tf.equal(tf.linalg.diag_part(H), 0)
            H = tf.where(tf.expand_dims(dead, 0), tf.ones_like(H), H)
            # Don't zero out the weights - this breaks quantization
            # W = tf.where(tf.expand_dims(dead, 0), tf.zeros_like(W), W)

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = tf.argsort(tf.linalg.diag_part(H), direction='DESCENDING')
            W = tf.gather(W, perm, axis=1)
            H = tf.gather(tf.gather(H, perm, axis=0), perm, axis=1)
            invperm = tf.argsort(perm)

        Losses = tf.zeros_like(W)
        Q = tf.zeros_like(W)
        Err = tf.zeros_like(W)

        damp = percdamp * tf.reduce_mean(tf.linalg.diag_part(H))
        # diag = tf.range(self.columns)
        # H = tf.tensor_scatter_nd_add(H, tf.expand_dims(diag, 1), tf.fill([self.columns], damp))
        H = tf.linalg.set_diag(H, tf.linalg.diag_part(H) + damp)
        H = tf.linalg.cholesky(H)
        H = tf.linalg.cholesky_solve(H, tf.eye(self.columns, dtype=tf.float32))
        H = tf.linalg.cholesky(H)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = tf.identity(W[:, i1:i2])
            Q1 = tf.zeros_like(W1)
            Err1 = tf.zeros_like(W1)
            Losses1 = tf.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                # Use quantize function from quantkeras
                from quantkeras import quantize
                # print(f"Quantizing column {i}: w range [{tf.reduce_min(w):.6f}, {tf.reduce_max(w):.6f}]")
                # print(f"Scale: {self.quantizer.scale}, Zero: {self.quantizer.zero}, Maxq: {self.quantizer.maxq}")
                q = quantize(
                    tf.expand_dims(w, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                )
                q = tf.squeeze(q)
                # print(f"Quantized q range [{tf.reduce_min(q):.6f}, {tf.reduce_max(q):.6f}]")
                indices = tf.stack([tf.range(Q1.shape[0]), tf.fill([Q1.shape[0]], i)], axis=1)
                Q1 = tf.tensor_scatter_nd_update(Q1, indices, q)
                Losses1 = tf.tensor_scatter_nd_update(Losses1, indices, tf.square(w - q) / (d ** 2))
                err1 = (w - q) / d
                # Only update the slice W1[:, i:]
                W1_slice = W1[:, i:] - tf.expand_dims(err1, 1) * Hinv1[i, i:]
                W1 = tf.concat([W1[:, :i], W1_slice], axis=1)
                Err1 = tf.tensor_scatter_nd_update(Err1, indices, err1)

            Q = tf.concat([Q[:, :to_python_int(i1)], Q1, Q[:, to_python_int(i2):]], axis=1)
            Losses = tf.concat([Losses[:, :to_python_int(i1)], Losses1 / 2, Losses[:, to_python_int(i2):]], axis=1)
            Err = tf.concat([Err[:, :to_python_int(i1)], Err1, Err[:, to_python_int(i2):]], axis=1)

            W_right = W[:, i2:] - tf.matmul(Err1, Hinv[i1:i2, i2:])
            W = tf.concat([W[:, :i2], W_right], axis=1)

            if DEBUG:
                self.layer.weights[0].assign(tf.concat([Q[:, :i2], W[:, i2:]], axis=1))
                print(tf.reduce_sum(tf.square(self.layer(self.inp1) - self.out1)))
                print(tf.reduce_sum(Losses))

        print('time %.2f' % (time.time() - tick))
        print('error', tf.reduce_sum(Losses).numpy())

        if actorder:
            Q = tf.gather(Q, invperm, axis=1)

        # Note: No Conv1D equivalent in Keras, so we skip that transpose
        # After quantization logic, before assignment
        print("Q before assignment (first 5):", Q.numpy().flatten()[:5])
        print("Q shape before assignment:", Q.shape)
        print("Original kernel shape:", self.layer.kernel.shape)
        # Ensure Q is 2D and matches kernel shape
        if len(Q.shape) != 2:
            Q = tf.reshape(Q, self.layer.kernel.shape)
        elif Q.shape != self.layer.kernel.shape:
            Q = tf.reshape(Q, self.layer.kernel.shape)
        self.layer.kernel.assign(tf.convert_to_tensor(Q, dtype=self.layer.kernel.dtype))
        if DEBUG:
            print(tf.reduce_sum(tf.square(self.layer(self.inp1) - self.out1)))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None 