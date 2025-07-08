import numpy as np
import tensorflow as tf
from tensorflow import keras

ops = tf  # Keras 3.0 ops API

# Quantize function for Keras ops

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return ops.cast(x > scale / 2, 'float32') * scale + ops.cast(x < zero / 2, 'float32') * zero
    q = tf.clip_by_value(tf.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer:
    def __init__(self, shape=1):
        self.maxq = ops.convert_to_tensor(0, dtype='float32')
        self.scale = ops.zeros(shape, dtype='float32')
        self.zero = ops.zeros(shape, dtype='float32')
        self.perchannel = False
        self.sym = True
        self.mse = False
        self.norm = 2.4
        self.grid = 100
        self.maxshrink = 0.8

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8, trits=False):
        self.maxq = ops.convert_to_tensor(2 ** bits - 1, dtype='float32')
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = ops.convert_to_tensor(-1, dtype='float32')

    def find_params(self, x, weight=False):
        shape = x.shape
        if self.perchannel:
            if weight:
                x = ops.reshape(x, [x.shape[0], -1])
            else:
                if len(shape) == 4:
                    x = ops.transpose(x, [1, 0, 2, 3])
                    x = ops.reshape(x, [x.shape[0], -1])
                if len(shape) == 3:
                    x = ops.transpose(ops.reshape(x, [-1, shape[-1]]), [1, 0])
                if len(shape) == 2:
                    x = ops.transpose(x)
        else:
            x = ops.reshape(x, [1, -1])

        tmp = ops.zeros([x.shape[0]], dtype=x.dtype)
        xmin = ops.minimum(tf.reduce_min(x, axis=1), tmp)
        xmax = ops.maximum(tf.reduce_max(x, axis=1), tmp)

        if self.sym:
            xmax = ops.maximum(ops.abs(xmin), xmax)
            tmp_mask = xmin < 0
            xmin = ops.where(tmp_mask, -xmax, xmin)
        tmp_mask = ops.logical_and(xmin == 0, xmax == 0)
        xmin = ops.where(tmp_mask, -ops.ones_like(xmin), xmin)
        xmax = ops.where(tmp_mask, ops.ones_like(xmax), xmax)

        # Fix: Use tf.reduce_all and tf.less for TensorFlow compatibility
        if tf.reduce_all(tf.less(self.maxq, 0)):
            scale = xmax
            zero = xmin
        else:
            scale = (xmax - xmin) / self.maxq
            if self.sym:
                zero = ops.ones_like(scale) * ((self.maxq + 1) / 2)
            else:
                zero = ops.round(-xmin / scale)

        if self.mse:
            best = tf.fill([x.shape[0]], float('inf'))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = ops.round(-xmin1 / scale1) if not self.sym else zero
                q = quantize(x, ops.expand_dims(scale1, 1), ops.expand_dims(zero1, 1), self.maxq)
                q = ops.abs(q - x)
                q = ops.pow(q, self.norm)
                err = tf.reduce_sum(q, axis=1)
                tmp_mask = err < best
                best = ops.where(tmp_mask, err, best)
                scale = ops.where(tmp_mask, scale1, scale)
                zero = ops.where(tmp_mask, zero1, zero)

        if not self.perchannel:
            if weight:
                rep = shape[0]
            else:
                rep = shape[1] if len(shape) != 3 else shape[2]
            scale = ops.repeat(scale, rep)
            zero = ops.repeat(zero, rep)

        if weight:
            new_shape = [-1] + [1] * (len(shape) - 1)
            scale = ops.reshape(scale, new_shape)
            zero = ops.reshape(zero, new_shape)
            self.scale = scale
            self.zero = zero
            return
        if len(shape) == 4:
            self.scale = ops.reshape(scale, [1, -1, 1, 1])
            self.zero = ops.reshape(zero, [1, -1, 1, 1])
        elif len(shape) == 3:
            self.scale = ops.reshape(scale, [1, 1, -1])
            self.zero = ops.reshape(zero, [1, 1, -1])
        elif len(shape) == 2:
            self.scale = ops.expand_dims(scale, 0)
            self.zero = ops.expand_dims(zero, 0)
        else:
            self.scale = scale
            self.zero = zero

    def quantize_tensor(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return tf.reduce_all(self.maxq > 0)

    def ready(self):
        return tf.reduce_all(self.scale != 0)