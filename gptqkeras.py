import math
import time
import tensorflow as tf
import keras

ops = tf  # Keras 3.0 ops API

DEBUG = False

class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        W = ops.convert_to_tensor(layer.weights[0].numpy())
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = ops.zeros((self.columns, self.columns), dtype='float32')
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = ops.expand_dims(inp, 0)
        tmp = inp.shape[0]
        if isinstance(self.layer, keras.layers.Dense):
            if len(inp.shape) == 3:
                inp = ops.reshape(inp, (-1, inp.shape[-1]))
            inp = ops.transpose(inp)
        self.H = self.H * (self.nsamples / (self.nsamples + tmp))
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * ops.cast(inp, 'float32')
        self.H = self.H + ops.matmul(inp, ops.transpose(inp))

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False):
        W = ops.convert_to_tensor(self.layer.weights[0].numpy(), dtype='float32')
        tick = time.time()

        if not hasattr(self, 'quantizer') or not getattr(self.quantizer, 'ready', lambda: False)():
            pass  # Quantizer logic placeholder

        H = self.H
        dead = ops.equal(tf.linalg.diag_part(H), 0)
        H = ops.where(ops.expand_dims(dead, 0), ops.ones_like(H), H)
        W = ops.where(ops.expand_dims(dead, 0), ops.zeros_like(W), W)

        if actorder:
            # Use tf.linalg.diag_part instead of ops.diagonal
            perm = tf.argsort(tf.linalg.diag_part(H), direction='DESCENDING')
            # Use tf.gather instead of ops.take
            W = tf.gather(W, perm, axis=1)
            H = tf.gather(tf.gather(H, perm, axis=0), perm, axis=1)
            invperm = tf.argsort(perm)

        Losses = tf.zeros_like(W)
        Q = ops.zeros_like(W)

        # Compute dampening value
        damp = percdamp * tf.reduce_mean(tf.linalg.diag_part(H))
        diag = tf.range(self.columns)
        # Add damp to diagonal
        H = tf.tensor_scatter_nd_add(H, tf.expand_dims(diag, 1), tf.fill([self.columns], damp))
        # Cholesky decomposition and inversion
        L = tf.linalg.cholesky(H)
        Hinv = tf.linalg.cholesky_solve(L, tf.eye(self.columns, dtype=tf.float32))
        H = Hinv  # For compatibility with rest of code
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
                q = w  # Quantizer logic placeholder

                # Update Q1: set column i to q
                Q1 = tf.tensor_scatter_nd_update(Q1, tf.expand_dims(tf.range(Q1.shape[0]), 1), tf.expand_dims(q, 1)) if Q1.shape[1] == 1 else tf.concat([Q1[:, :i], tf.expand_dims(q, 1), Q1[:, i+1:]], axis=1)

                # Update Losses1: set column i
                loss_val = tf.square(w - q) / (d ** 2)
                Losses1 = tf.tensor_scatter_nd_update(Losses1, tf.expand_dims(tf.range(Losses1.shape[0]), 1), tf.expand_dims(loss_val, 1)) if Losses1.shape[1] == 1 else tf.concat([Losses1[:, :i], tf.expand_dims(loss_val, 1), Losses1[:, i+1:]], axis=1)

                err1 = (w - q) / d

                # Update W1: set column i
                update_val = tf.matmul(tf.expand_dims(err1, 1), tf.expand_dims(Hinv1[i, i:], 0))
                W1 = tf.concat([W1[:, :i], update_val, W1[:, i+1:]], axis=1) if W1.shape[1] > 1 else update_val

                # Update Err1: set column i
                # Update Err1: set column i
                Err1 = tf.concat([Err1[:, :i], tf.expand_dims(err1, 1), Err1[:, i+1:]], axis=1)

                # Update Q and Losses using tensor_scatter_nd_update instead of ops.update
                # Q: update columns i1:i2 with Q1
                Q = tf.concat([Q[:, :i1], Q1, Q[:, i2:]], axis=1)
                # Losses: update columns i1:i2 with Losses1 / 2
                Losses = tf.concat([Losses[:, :i1], Losses1 / 2, Losses[:, i2:]], axis=1)
                # W: update columns i2: with tf.matmul(Err1, Hinv[i1:i2, i2:])
                W = tf.concat([W[:, :i2], tf.matmul(Err1, Hinv[i1:i2, i2:])], axis=1)

                if DEBUG:
                    self.layer.weights[0].assign(tf.concat([Q[:, :i2], W[:, i2:]], axis=1))
                    print(tf.reduce_sum(tf.square(self.layer(self.inp1) - self.out1)))
                    print(tf.reduce_sum(Losses))

        print('time %.2f' % (time.time() - tick))
        print('error', ops.sum(Losses))

        if actorder:
            Q = tf.gather(Q, invperm, axis=1)

        self.layer.weights[0].assign(tf.reshape(Q, self.layer.weights[0].shape))

        if DEBUG:
            print(tf.reduce_sum(tf.square(self.layer(self.inp1) - self.out1)))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None 