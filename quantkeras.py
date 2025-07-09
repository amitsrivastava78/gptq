import numpy as np
import tensorflow as tf
import keras

ops = tf  # Keras 3.0 ops API

# Quantize function for Keras ops (equivalent to PyTorch version)
def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return tf.cast(x > scale / 2, tf.float32) * scale + tf.cast(x < zero / 2, tf.float32) * zero
    q = tf.clip_by_value(tf.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

class Quantizer:
    def __init__(self, shape=1):
        # Equivalent to PyTorch's register_buffer
        self.maxq = tf.convert_to_tensor(0, dtype=tf.float32)
        self.scale = tf.zeros(shape, dtype=tf.float32)
        self.zero = tf.zeros(shape, dtype=tf.float32)

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = tf.convert_to_tensor(2 ** bits - 1, dtype=tf.float32)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = tf.convert_to_tensor(-1, dtype=tf.float32)

    def find_params(self, x, weight=False):
        # Get device (in TensorFlow this is handled automatically)
        shape = x.shape
        if self.perchannel:
            if weight:
                x = tf.reshape(x, [x.shape[0], -1])
            else:
                if len(shape) == 4:
                    x = tf.transpose(x, [1, 0, 2, 3])
                    x = tf.reshape(x, [x.shape[0], -1])
                if len(shape) == 3:
                    x = tf.transpose(tf.reshape(x, [-1, shape[-1]]), [1, 0])
                if len(shape) == 2:
                    x = tf.transpose(x)
        else:
            x = tf.reshape(x, [1, -1])

        tmp = tf.zeros([x.shape[0]], dtype=x.dtype)
        xmin = tf.minimum(tf.reduce_min(x, axis=1), tmp)
        xmax = tf.maximum(tf.reduce_max(x, axis=1), tmp)

        if self.sym:
            xmax = tf.maximum(tf.abs(xmin), xmax)
            tmp_mask = xmin < 0
            if tf.reduce_any(tmp_mask):
                xmin = tf.where(tmp_mask, -xmax, xmin)
        tmp_mask = tf.logical_and(tf.equal(xmin, 0), tf.equal(xmax, 0))
        xmin = tf.where(tmp_mask, -tf.ones_like(xmin), xmin)
        xmax = tf.where(tmp_mask, tf.ones_like(xmax), xmax)

        if tf.less(self.maxq, 0):
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = tf.fill(tf.shape(self.scale), tf.add(self.maxq, 1) / 2)
            else:
                self.zero = tf.round(-xmin / self.scale)

        if self.mse:
            best = tf.fill([x.shape[0]], float('inf'))
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = tf.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, tf.expand_dims(scale1, 1), tf.expand_dims(zero1, 1), self.maxq)
                q = q - x
                q = tf.abs(q)
                q = tf.pow(q, self.norm)
                err = tf.reduce_sum(q, axis=1)
                tmp_mask = err < best
                if tf.reduce_any(tmp_mask):
                    best = tf.where(tmp_mask, err, best)
                    self.scale = tf.where(tmp_mask, scale1, self.scale)
                    self.zero = tf.where(tmp_mask, zero1, self.zero)
        
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = tf.repeat(self.scale, tmp)
            self.zero = tf.repeat(self.zero, tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = tf.reshape(self.scale, shape)
            self.zero = tf.reshape(self.zero, shape)
            return
        if len(shape) == 4:
            self.scale = tf.reshape(self.scale, (1, -1, 1, 1))
            self.zero = tf.reshape(self.zero, (1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = tf.reshape(self.scale, (1, 1, -1))
            self.zero = tf.reshape(self.zero, (1, 1, -1)) 
        if len(shape) == 2:
            self.scale = tf.expand_dims(self.scale, 0)
            self.zero = tf.expand_dims(self.zero, 0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return tf.reduce_all(tf.greater(self.maxq, 0))

    def ready(self):
        return tf.reduce_all(tf.not_equal(self.scale, 0))