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
        input_dim = int(W.shape[0])
        output_dim = int(W.shape[1])
        self.H = tf.zeros((output_dim, output_dim), dtype=tf.float32)
        # print(f"The HESSAIN MATRIX shape is {self.H.shape}")
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
        if inp is None or out is None:
            print("add_batch received None input or output, skipping.")
            return
        # print("Inside GPTQ add_batch")
        # print("Input shape:", inp.shape)
        # print("Output shape:", out.shape)
        
        # For Keras Dense layers, we want to accumulate the Hessian over the OUTPUT dimension
        # The Hessian should be (output_dim, output_dim)
        
        # 1. Reshape 3D outputs to 2D. This leaves 2D outputs unchanged.
        if len(out.shape) == 3:
            out = tf.reshape(out, [-1, out.shape[-1]])  # [batch*seq, output_features]
        
        # 2. Transpose to get (output_features, batch*seq)
        out = tf.transpose(out)  # [output_features, batch*seq]
        num_new_samples = out.shape[1]  # number of columns = number of samples
        
        # print("self.H shape:", self.H.shape)
        # print("out shape:", out.shape)
        # print("matmul shape:", tf.matmul(out, tf.transpose(out)).shape)
        
        # 3. Update Hessian with running average
        self.H = self.H * (self.nsamples / (self.nsamples + num_new_samples))
        self.nsamples += num_new_samples
        # print(f"SAMLPLE value is {self.nsamples}")
        
        # 4. Scale and accumulate
        out = tf.sqrt(2.0 / tf.cast(self.nsamples, tf.float32)) * out
        self.H = self.H + tf.matmul(out, tf.transpose(out))  # [output_features, output_features]

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
            # Add numerical stability checks
            dead = tf.equal(tf.linalg.diag_part(H), 0)
            H = tf.where(tf.expand_dims(dead, 0), tf.ones_like(H), H)
            
            # Check for NaN or Inf in Hessian
            if tf.reduce_any(tf.math.is_nan(H)) or tf.reduce_any(tf.math.is_inf(H)):
                print("WARNING: NaN/Inf detected in Hessian. Using identity matrix.")
                H = tf.eye(self.columns, dtype=tf.float32)

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

        # More robust damping for CPU
        damp = percdamp * tf.reduce_mean(tf.linalg.diag_part(H))
        # Ensure minimum damping for numerical stability
        min_damp = 1e-6
        damp = tf.maximum(damp, min_damp)
        
        H = tf.linalg.set_diag(H, tf.linalg.diag_part(H) + damp)
        
        # Robust Cholesky decomposition with fallback
        try:
            # Try Cholesky decomposition
            H_chol = tf.linalg.cholesky(H)
            Hinv = tf.linalg.cholesky_solve(H_chol, tf.eye(self.columns, dtype=tf.float32))
        except Exception as e:
            print(f"Cholesky decomposition failed: {e}. Using pseudo-inverse.")
            # Fallback to pseudo-inverse
            try:
                Hinv = tf.linalg.pinv(H)
            except Exception as e2:
                print(f"Pseudo-inverse also failed: {e2}. Using identity matrix.")
                Hinv = tf.eye(self.columns, dtype=tf.float32)
        
        # Check for numerical issues in inverse
        if tf.reduce_any(tf.math.is_nan(Hinv)) or tf.reduce_any(tf.math.is_inf(Hinv)):
            print("WARNING: NaN/Inf in Hessian inverse. Using identity matrix.")
            Hinv = tf.eye(self.columns, dtype=tf.float32)

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
                
                # Check for numerical issues
                if tf.math.is_nan(d) or tf.math.is_inf(d) or tf.abs(d) < 1e-10:
                    print(f"WARNING: Invalid diagonal element at {i1+i}. Skipping quantization.")
                    # Just copy the original weight
                    indices = tf.stack([tf.range(Q1.shape[0]), tf.fill([Q1.shape[0]], i)], axis=1)
                    Q1 = tf.tensor_scatter_nd_update(Q1, indices, w)
                    continue

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
                try:
                    q = quantize(
                        tf.expand_dims(w, 1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    )
                    q = tf.squeeze(q)
                    
                    # Check for NaN in quantized values
                    if tf.reduce_any(tf.math.is_nan(q)):
                        print(f"WARNING: NaN in quantized values at {i1+i}. Using original weights.")
                        q = w
                        
                except Exception as e:
                    print(f"Quantization failed at {i1+i}: {e}. Using original weights.")
                    q = w
                
                indices = tf.stack([tf.range(Q1.shape[0]), tf.fill([Q1.shape[0]], i)], axis=1)
                Q1 = tf.tensor_scatter_nd_update(Q1, indices, q)
                Losses1 = tf.tensor_scatter_nd_update(Losses1, indices, tf.square(w - q) / (d ** 2))
                err1 = (w - q) / d
                
                # Check for numerical issues in error
                if tf.reduce_any(tf.math.is_nan(err1)) or tf.reduce_any(tf.math.is_inf(err1)):
                    print(f"WARNING: NaN/Inf in error at {i1+i}. Skipping weight update.")
                    continue
                    
                # Only update the slice W1[:, i:]
                try:
                    W1_slice = W1[:, i:] - tf.expand_dims(err1, 1) * Hinv1[i, i:]
                    # Check for NaN in updated weights
                    if tf.reduce_any(tf.math.is_nan(W1_slice)):
                        print(f"WARNING: NaN in weight update at {i1+i}. Skipping update.")
                    else:
                        W1 = tf.concat([W1[:, :i], W1_slice], axis=1)
                except Exception as e:
                    print(f"Weight update failed at {i1+i}: {e}. Continuing.")

            # Update the main weight matrix
            W = tf.concat([W[:, :i1], Q1, W[:, i2:]], axis=1)

        if actorder:
            W = tf.gather(W, invperm, axis=1)

        # Update the layer weights
        try:
            self.layer.weights[0].assign(W)
        except Exception as e:
            print(f"Failed to assign weights: {e}")

        print('time %.2f' % (time.time() - tick))
        print('error', tf.reduce_mean(Losses).numpy())

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None 