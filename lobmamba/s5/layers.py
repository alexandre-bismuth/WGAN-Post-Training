from flax import linen as nn
import jax

from s5 import es_fold


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity,
            dropout, batch/layer norm, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            dropout     (float32):  dropout rate
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            training    (bool):     whether in training mode or not
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    dropout: float
    d_model: int
    activation: str = "gelu"
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.90
    step_rescale: float = 1.0

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout
        """

        self.seq = self.ssm(step_rescale=self.step_rescale)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        if self.batchnorm:
            self.norm = nn.BatchNorm(use_running_average=not self.training,
                                     momentum=self.bn_momentum, axis_name='batch')
        else:
            self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

    def __call__(self, x, es=None):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
             es: optional EGGROLL LoRA factors for this layer (None = stock path)
        Returns:
            output sequence (float32): (L, d_model)
        """
        es_seq = es_fold.subtree(es, "seq")

        def _out2(v):
            return es_fold.fold_kernel(es, "out2", v, self.out2(v))

        skip = x
        if self.prenorm:
            x = self.norm(x)

        #jax.debug.print("call x before ssm : {}",x)
        # es kwarg only when present: non-Mamba SSM classes keep their stock signature.
        x = self.seq(x) if es_seq is None else self.seq(x, es=es_seq)
        #jax.debug.print("call x_m after ssm : {}",x)
        if self.activation in ["full_glu"]:
            x = self.drop(nn.gelu(x))
            x = self.out1(x) * jax.nn.sigmoid(_out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu1"]:
            x = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(_out2(x))
            x = self.drop(x)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(nn.gelu(x))
            x = x * jax.nn.sigmoid(_out2(x1))
            x = self.drop(x)
        elif self.activation in ["gelu"]:
            x = self.drop(nn.gelu(x))
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        #jax.debug.print("call x_m[0:5] after activation : {}",x[0:2][0][0:2])


        x = skip + x
        if not self.prenorm:
            x = self.norm(x)

        return x

    def __call_rnn__(self,hidden, x,d, es=None):
            """
            Compute the LxH output of the layer given an LxH input.
            Args:
                hidden : hidden state
                x (float32): input sequence (L, d_model)
                d (bool): reset signal (L,)
                es: optional EGGROLL LoRA factors for this layer (None = stock path)
            Returns:
                output sequence (float32): (L, d_model)
            """
            es_seq = es_fold.subtree(es, "seq")

            def _out2(v):
                return es_fold.fold_kernel(es, "out2", v, self.out2(v))

            skip = x
            if self.prenorm:
                x = self.norm(x)

            # es kwarg only when present: non-Mamba SSM classes keep their stock signature.
            if es_seq is None:
                hidden,x = self.seq.__call_rnn__(hidden,x,d)
            else:
                hidden,x = self.seq.__call_rnn__(hidden,x,d, es=es_seq)
            #hidden, x = jax.vmap(self.seq.__call_rnn__, in_axes=(None,1,None), out_axes=1)(hidden, x, d)


            if self.activation in ["full_glu"]:
                x = self.drop(nn.gelu(x))
                x = self.out1(x) * jax.nn.sigmoid(_out2(x))
                x = self.drop(x)
            elif self.activation in ["half_glu1"]:
                x = self.drop(nn.gelu(x))
                x = x * jax.nn.sigmoid(_out2(x))
                x = self.drop(x)
            elif self.activation in ["half_glu2"]:
                # Only apply GELU to the gate input
                x1 = self.drop(nn.gelu(x))
                x = x * jax.nn.sigmoid(_out2(x1))
                x = self.drop(x)
            elif self.activation in ["gelu"]:
                x = self.drop(nn.gelu(x))
            else:
                raise NotImplementedError(
                    "Activation: {} not implemented".format(self.activation))

            x = skip + x
            if not self.prenorm:
                x = self.norm(x)

            return hidden, x
    @staticmethod
    def initialize_carry(batch_size, hidden_size, ssm_type='gdn', **kwargs):
        if ssm_type == 'mamba3':
            # Mamba3 carry: (angle_state, ssm_state, k_state, v_state)
            nh = kwargs['n_heads']
            hd = kwargs['headdim']
            N = kwargs['d_state']
            num_rope_angles = kwargs.get('num_rope_angles', N // 4)
            angle_state = jax.numpy.zeros((batch_size, 1, nh, num_rope_angles), dtype=jax.numpy.float32)
            ssm_state = jax.numpy.zeros((batch_size, 1, nh, hd, N), dtype=jax.numpy.float32)
            k_state = jax.numpy.zeros((batch_size, 1, 1, nh, N), dtype=jax.numpy.float32)
            v_state = jax.numpy.zeros((batch_size, 1, nh, hd), dtype=jax.numpy.float32)
            return (angle_state, ssm_state, k_state, v_state)
        else:
            # GDN carry: matrix state S + optional Conv1d buffers
            nh = kwargs.get('num_heads', kwargs.get('n_heads'))
            hd = kwargs.get('head_dim', kwargs.get('headdim'))
            hvd = kwargs.get('head_v_dim', hd * kwargs.get('expand_v', 2))
            use_conv = kwargs.get('use_conv', True)
            conv_k = kwargs.get('conv_kernel_size', 4)
            S = jax.numpy.zeros((batch_size, 1, nh, hvd, hd), dtype=jax.numpy.float32)
            if use_conv:
                q_buf = jax.numpy.zeros((batch_size, conv_k - 1, nh * hd), dtype=jax.numpy.float32)
                k_buf = jax.numpy.zeros((batch_size, conv_k - 1, nh * hd), dtype=jax.numpy.float32)
                v_buf = jax.numpy.zeros((batch_size, conv_k - 1, nh * hvd), dtype=jax.numpy.float32)
                return (S, (q_buf, k_buf, v_buf))
            return S