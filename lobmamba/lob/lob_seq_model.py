import os
from functools import partial
from typing import Tuple
import jax
import jax.numpy as jnp
from flax import linen as nn
from s5.layers import SequenceLayer
from s5.seq_model import StackedEncoderModel, masked_meanpool
from s5 import es_fold
from lob.encoding_1tok import FIELD_VOCAB_SIZES_WITH_SPECIAL


class LobPredModel(nn.Module):
    """ S5 classificaton sequence model. This consists of the stacked encoder
    (which consists of a linear encoder and stack of S5 layers), mean pooling
    across the sequence length, a linear decoder, and a softmax operation.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_output     (int32):    the output dimension, i.e. the number of classes
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                        we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            padded:     (bool):     if true: padding was used
            activation  (string):   Type of activation function to use
            dropout     (float32):  dropout rate
            training    (bool):     whether in training mode or not
            mode        (str):      Options: [pool: use mean pooling, last: just take
                                                                       the last state]
            prenorm     (bool):     apply prenorm if true or postnorm if false
            batchnorm   (bool):     apply batchnorm if true or layernorm if false
            bn_momentum (float32):  the batchnorm momentum if batchnorm is used
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    padded: bool
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoderModel(
                            ssm=self.ssm,
                            d_model=self.d_model,
                            n_layers=self.n_layers,
                            activation=self.activation,
                            dropout=self.dropout,
                            training=self.training,
                            prenorm=self.prenorm,
                            batchnorm=self.batchnorm,
                            bn_momentum=self.bn_momentum,
                            step_rescale=self.step_rescale,
                                        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x.astype(jnp.float32), axis=-1)

    def __call_rnn__(self,hidden, x,d, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder.__call_rnn__(hidden, x,d, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError("Mode must be in ['pool'] for self.padded=True (for now...)")
            else:
                x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x.astype(jnp.float32), axis=-1)

# Here we call vmap to parallelize across a batch of input sequences
BatchLobPredModel = nn.vmap(
    LobPredModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')


class LobBookModel(nn.Module):
    ssm: nn.Module
    d_book: int
    d_model: int
    #n_layers: int
    n_pre_layers: int
    n_post_layers: int
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes ...
        """
        LayerCls = SequenceLayer
        if os.environ.get('REMAT', '0') == '1':
            policy = None
            if os.environ.get('REMAT_POLICY', '') == 'dots':
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            LayerCls = nn.remat(SequenceLayer, policy=policy)

        self.pre_layers = tuple(
            LayerCls(
                # fix ssm init to correct shape (different than other layers)
                ssm=partial(self.ssm, H=self.d_book),
                dropout=self.dropout,
                d_model=self.d_book,  # take book series as is
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
            ) for _ in range(self.n_pre_layers))
        self.projection = nn.Dense(self.d_model)  # project to d_model
        self.post_layers = tuple(
            LayerCls(
                ssm=self.ssm,
                dropout=self.dropout,
                d_model=self.d_model,
                activation=self.activation,
                training=self.training,
                prenorm=self.prenorm,
                batchnorm=self.batchnorm,
                bn_momentum=self.bn_momentum,
                step_rescale=self.step_rescale,
            )
            for _ in range(self.n_post_layers)
        )

    def __call__(self, x, integration_timesteps, es=None):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
             es: optional EGGROLL LoRA factors for this module (None = stock path)
        Returns:
            output sequence (float32): (L, d_model)
        """
        for i, layer in enumerate(self.pre_layers):
            es_i = es_fold.subtree(es, f"pre_layers_{i}")
            x = layer(x) if es_i is None else layer(x, es=es_i)
        x = es_fold.fold_kernel(es, "projection", x, self.projection(x))
        for i, layer in enumerate(self.post_layers):
            es_i = es_fold.subtree(es, f"post_layers_{i}")
            x = layer(x) if es_i is None else layer(x, es=es_i)
        return x

    def __call_rnn__(self, hiddens, x, d,integration_timesteps, es=None):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
             es: optional EGGROLL LoRA factors for this module (None = stock path)
        Returns:
            output sequence (float32): (L, d_model)
        """
        hidden_pre,hidden_post=hiddens
        new_hiddens_pre,new_hiddens_post = [],[]

        for i, layer in enumerate(self.pre_layers):
            es_i = es_fold.subtree(es, f"pre_layers_{i}")
            if es_i is None:
                new_h,x = layer.__call_rnn__(hidden_pre[i],x,d)
            else:
                new_h,x = layer.__call_rnn__(hidden_pre[i],x,d, es=es_i)
            new_hiddens_pre.append(new_h)

        x = es_fold.fold_kernel(es, "projection", x, self.projection(x))

        for i, layer in enumerate(self.post_layers):
            es_i = es_fold.subtree(es, f"post_layers_{i}")
            if es_i is None:
                new_h,x = layer.__call_rnn__(hidden_post[i],x,d)
            else:
                new_h,x = layer.__call_rnn__(hidden_post[i],x,d, es=es_i)
            new_hiddens_post.append(new_h)

        return (new_hiddens_pre,new_hiddens_post),x
    @staticmethod
    def initialize_carry(batch_size, hidden_size, n_layers_pre, n_layers_post, **gdn_kwargs):
        # Book pre-layers use H=d_book which auto-adjusts head count
        ssm_type = gdn_kwargs.get('ssm_type', 'gdn')
        pre_gdn_kwargs = gdn_kwargs
        if 'd_book' in gdn_kwargs:
            d_book = gdn_kwargs['d_book']
            if ssm_type == 'mamba3':
                hd = gdn_kwargs['headdim']
                expand = gdn_kwargs.get('expand', 2)
                d_inner_target = expand * d_book
                eff_hd = min(hd, d_inner_target)
                eff_nh = max(1, d_inner_target // eff_hd)
                pre_gdn_kwargs = dict(gdn_kwargs, n_heads=eff_nh, headdim=eff_hd)
            else:
                hd = gdn_kwargs['head_dim']
                nh = gdn_kwargs['num_heads']
                eff_nh = min(nh, max(1, d_book // hd))
                eff_hd = min(hd, d_book)
                eff_hvd = eff_hd * (gdn_kwargs['head_v_dim'] // hd)
                pre_gdn_kwargs = dict(gdn_kwargs, num_heads=eff_nh,
                                      head_dim=eff_hd, head_v_dim=eff_hvd)
        init_hidden = (
            [SequenceLayer.initialize_carry(batch_size, hidden_size, **pre_gdn_kwargs)
             for _ in range(n_layers_pre)],
            [SequenceLayer.initialize_carry(batch_size, hidden_size, **gdn_kwargs)
             for _ in range(n_layers_post)],
        )
        return init_hidden
    
    

class FullLobPredModel(nn.Module):
    ssm: nn.Module
    d_output: int
    d_model: int
    d_book: int
    n_message_layers: int
    n_fused_layers: int
    n_book_pre_layers: int = 1
    n_book_post_layers: int = 1
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        self.message_encoder = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_message_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=True,
            vocab_size=self.d_output,
        )
        # applied to transposed message output to get seq len for fusion
        self.message_out_proj = nn.Dense(self.d_model)  
        self.book_encoder = LobBookModel(
            ssm=self.ssm,
            d_book=self.d_book,
            d_model=self.d_model,
            n_pre_layers=self.n_book_pre_layers,
            n_post_layers=self.n_book_post_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        # applied to transposed book output to get seq len for fusion
        self.book_out_proj = nn.Dense(self.d_model)
        self.fused_s5 = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_fused_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        (L_m x d_input, L_b x [P+1]) input sequence tuple,
        combining message and book inputs.
        Args:
             x (float32): 2-tuple of input sequences (L_m x d_input, L_b x [P+1])
        Returns:
            output (float32): (d_output)
        """
        #x_m, x_b = x
        # print(x_m.shape, x_b.shape)

        x_m = self.message_encoder(x_m, message_integration_timesteps)
        # TODO: check integration time steps make sense here
        x_b = self.book_encoder(x_b, book_integration_timesteps)

        x_m = self.message_out_proj(x_m.T).T
        x_b = self.book_out_proj(x_b.T).T
        x = jnp.concatenate([x_m, x_b], axis=1)
        # TODO: again, check integration time steps make sense here
        x = self.fused_s5(x, jnp.ones(x.shape[0]))

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x.astype(jnp.float32), axis=-1)



# Here we call vmap to parallelize across a batch of input sequences
BatchFullLobPredModel = nn.vmap(
    FullLobPredModel,
    in_axes=(0, 0, 0, 0),
    out_axes=0,
    variable_axes={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None},
    split_rngs={"params": False, "dropout": True}, axis_name='batch')

## Repeat shorter sequences, instead of linear projection:

class PaddedLobPredModel(nn.Module):
    ssm: nn.Module
    d_output: int
    d_model: int
    d_book: int
    n_message_layers: int
    n_fused_layers: int
    n_book_pre_layers: int = 1
    n_book_post_layers: int = 1
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0
    # MoE parameters (only applied to fused_s5)

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        # nn.checkpoint()
        self.message_encoder = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_message_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=True,
            vocab_size=self.d_output,
        )

        # applied to transposed message output to get seq len for fusion
        #self.message_out_proj = nn.Dense(self.d_model)  
        # nn.checkpoint()
        self.book_encoder = LobBookModel(
            ssm=self.ssm,
            d_book=self.d_book,
            d_model=self.d_model,
            n_pre_layers=self.n_book_pre_layers,
            n_post_layers=self.n_book_post_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )


        # applied to transposed book output to get seq len for fusion
        #self.book_out_proj = nn.Dense(self.d_model)
        # nn.checkpoint()

        self.fused_s5 = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_fused_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        (L_m x d_input, L_b x [P+1]) input sequence tuple,
        combining message and book inputs.
        Args:
             x_m: message input sequence (L_m x d_input,
             x_b: book state (volume series) (L_b x [P+1])
        Returns:
            output (float32): (d_output)
        """
        # jax.debug.print("x_m shape: {}",x_m.shape)

        #x_b = jnp.repeat(x_b, x_m.shape[0] // x_b.shape[0], axis=0)
        # jax.debug.print("call x_m[0:5] before msg_enc : {}",x_m[0:5])

        x_m = self.message_encoder(x_m, message_integration_timesteps)
        # jax.debug.print("call x_m[0:5] after msg_enc : {}",x_m[0:5][0])
        x_b = self.book_encoder(x_b, book_integration_timesteps)

        # repeat book input to match message length
        #Move this repeat to the dataloading and edit so that alignment works with shifted tokens. 
        #x_b = jnp.repeat(x_b, x_m.shape[0] // x_b.shape[0], axis=0)
        #REPEAT SHOULD NO LONGER BE NEEDED DUE TO REPEATING HAPPENING IN DATALOADER

        # token_index = 5 # TODO

        # # Calculate the repeat counts for each segment
        # K = x_m.shape[0] // x_b.shape[0] # TODO number of tokens in one message
        # repeats = [K - token_index] + [K] * (x_b.shape[0] - 2) + [token_index]
        # x_b = jnp.concatenate([jnp.repeat(x_b[i:i+1], repeats[i], axis=0) for i in range(x_b.shape[0])], axis=0)
        
            
        x = jnp.concatenate([x_m, x_b], axis=1)
        # TODO: again, check integration time steps make sense here
        x = self.fused_s5(x, jnp.ones(x.shape[0]))

        jax.debug.print("x output shape {}, 1st five: \n {}",x.shape,x[:5,:5])

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        elif self.mode in ["none"]:
            pass
        elif self.mode in ['ema']:
            x,_=ewma_vectorized_safe(x,2 /(24 + 1.0),jnp.zeros((1,x.shape[1])),jnp.array(1))
            #FIXME: Provide the ntoks argument for averaging as an arg.
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last','none','ema']")
        
        jax.debug.print("x output shape after pool/last/ema/none shape {}, 1st five: \n {}",x.shape,x[:5,:5])
        x = self.decoder(x)
        jax.debug.print("x output shape after decoder {}",x.shape,x[:5,:5])

        return nn.log_softmax(x.astype(jnp.float32), axis=-1)


    #FOR AR version....
    def __call_rnn__(self,hiddens_tuple,
                      x_m, x_b,
                      d_m, d_b,
                      d_f,
                      message_integration_timesteps, book_integration_timesteps,
                      es=None):
        """
        FOR full output version
        Compute the size d_output log softmax output given a
        (L_m x d_input, L_b x [P+1]) input sequence tuple,
        combining message and book inputs.
        Args:
             x_m: message input sequence (L_m x d_input,
             x_b: book state (volume series) (L_b x [P+1])
             es: optional EGGROLL LoRA factors, mirroring the params tree at the
                 perturbed projection kernels only (None = stock path, byte-identical)
        Returns:
            output (float32): (d_output)
        """
        hiddens_m, hiddens_b,hiddens_fused,ema = hiddens_tuple
        fo,override=ema

        # print("Shapes:",x_m.shape,x_b.shape,d_m.shape,d_b.shape)

        # print("Shapes:",x_m.shape,x_b.shape,d_m.shape,d_b.shape)

        es_m = es_fold.subtree(es, "message_encoder")
        es_b = es_fold.subtree(es, "book_encoder")
        es_f = es_fold.subtree(es, "fused_s5")
        if es_m is None:
            hiddens_m,x_m = self.message_encoder.__call_rnn__(hiddens_m, x_m,d_m, message_integration_timesteps)
        else:
            hiddens_m,x_m = self.message_encoder.__call_rnn__(hiddens_m, x_m,d_m, message_integration_timesteps, es=es_m)
        if es_b is None:
            hiddens_b,x_b = self.book_encoder.__call_rnn__(hiddens_b,x_b,d_b ,book_integration_timesteps)
        else:
            hiddens_b,x_b = self.book_encoder.__call_rnn__(hiddens_b,x_b,d_b ,book_integration_timesteps, es=es_b)
        x = jnp.concatenate([x_m, x_b], axis=1)
        # TODO: again, check integration time steps make sense here
        if es_f is None:
            hiddens_fused,x = self.fused_s5.__call_rnn__(hiddens_fused, x, d_f, jnp.ones(x.shape[0]))
        else:
            hiddens_fused,x = self.fused_s5.__call_rnn__(hiddens_fused, x, d_f, jnp.ones(x.shape[0]), es=es_f)

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        elif self.mode in ["none"]:
            pass
        elif self.mode in ['ema']:
             print("x",x)
             print("ema",ema)
             x,fo=ewma_vectorized_safe(x,2 /(24 + 1.0),fo,override)
        else:
            raise NotImplementedError("Must double check before running rnn")

        x = self.decoder(x)
        return (hiddens_m, hiddens_b, hiddens_fused, (fo,jnp.zeros_like(override))), nn.log_softmax(x.astype(jnp.float32), axis=-1)

    def __call_ar__(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        (L_m x d_input, L_b x [P+1]) input sequence tuple,
        combining message and book inputs.
        Args:
             x_m: message input sequence (L_m x d_input, 
             x_b: book state (volume series) (L_b x [P+1])
        Returns:
            output (float32): (d_output)
        """
        #Uncomment to debug if no longer working in data-loader. 

        x_m = self.message_encoder(x_m, message_integration_timesteps)
        x_b = self.book_encoder(x_b, book_integration_timesteps)

        #Works because book already repeated when loading data. 
        x = jnp.concatenate([x_m, x_b], axis=1)
        # TODO: again, check integration time steps make sense here
        x = self.fused_s5(x, jnp.ones(x.shape[0]))

        #Removed the pooling to enable each token to be a target,
        #  not just a random one in the last message. 

        # jax.debug.print("x output shape {}, 1st five: \n {}",x.shape,x[:5,:5])

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        elif self.mode in ["none"]:
            pass
        elif self.mode in ['ema']:
            x,_=ewma_vectorized_safe(x,2 /(24 + 1.0),jnp.zeros((1,x.shape[1])),jnp.array(1))
            #FIXME: Provide the ntoks argument for averaging as an arg.
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last','none','ema']")
        
        # jax.debug.print("x output shape after pool/last/ema/none shape {}, 1st five: \n {}",x.shape,x[:5,:5])
        x = self.decoder(x)
        # jax.debug.print("x output shape after decoder {}, 1st five: \n {}",x.shape,x[:5,:5])

        
        x = nn.log_softmax(x.astype(jnp.float32), axis=-1)
        return x
    
    @staticmethod
    def initialize_carry(batch_size, hidden_size,
                         n_message_layers,
                         n_book_pre_layers,
                         n_book_post_layers,
                         n_fused_layers,
                         h_size_ema,
                         **gdn_kwargs):
        # Use a dummy key since the default state init fn is just zeros.
        h_tuple_init = (
            StackedEncoderModel.initialize_carry(
                batch_size, hidden_size, n_message_layers, **gdn_kwargs),
            LobBookModel.initialize_carry(
                batch_size, hidden_size, n_book_pre_layers, n_book_post_layers, **gdn_kwargs),
            StackedEncoderModel.initialize_carry(
                batch_size, hidden_size, n_fused_layers, **gdn_kwargs),
            (jnp.zeros((batch_size, 1, h_size_ema)),
             jnp.ones((batch_size, 1, 1))),
        )
        return h_tuple_init

split_rngs_args={"params": False, "dropout": True}
variable_axes_args={"params": None, "dropout": None, 'batch_stats': None, "cache": 0, "prime": None}


# Here we call vmap to parallelize across a batch of input sequences
OldBatchPaddedLobPredModel = nn.vmap(
    PaddedLobPredModel,
    in_axes=(0, 0, 0, 0),
    out_axes=0,
    variable_axes=variable_axes_args,
    split_rngs=split_rngs_args, axis_name='batch',)

BatchPaddedLobPredModel = nn.vmap(
    PaddedLobPredModel,
    in_axes=(0, 0, 0, 0),
    out_axes=0,
    variable_axes=variable_axes_args,
    split_rngs=split_rngs_args, axis_name='batch',
    methods={'__call__':{'in_axes':(0, 0, 0, 0),
                         'out_axes':0,
                         'variable_axes':variable_axes_args,
                         'split_rngs':split_rngs_args,
                         'axis_name':'batch'},
            '__call_rnn__':{'in_axes':(0,0, 0, 0, 0, 0,0,0),
                         'out_axes':0,
                         'variable_axes':variable_axes_args,
                         'split_rngs':split_rngs_args,
                         'axis_name':'batch'},
            '__call_ar__':{'in_axes':(0, 0, 0, 0),
                         'out_axes':0,
                         'variable_axes':variable_axes_args,
                         'split_rngs':split_rngs_args,
                         'axis_name':'batch'}})


# EGGROLL ES variant (S5b): same module -> identical param tree -> same checkpoint.
# __call_rnn__ takes a 9th arg `es` (the LoRA factor pytree), broadcast (in_axes=None)
# across the INNER batch axis — the ES population axis is the OUTER jax.vmap in
# eggroll_gan.es_generator, which maps `es` with in_axes=0 so each direction's
# (A_i, B_i) stays aligned with its own activations.
BatchPaddedLobPredModelES = nn.vmap(
    PaddedLobPredModel,
    in_axes=(0, 0, 0, 0),
    out_axes=0,
    variable_axes=variable_axes_args,
    split_rngs=split_rngs_args, axis_name='batch',
    methods={'__call__':{'in_axes':(0, 0, 0, 0),
                         'out_axes':0,
                         'variable_axes':variable_axes_args,
                         'split_rngs':split_rngs_args,
                         'axis_name':'batch'},
            '__call_rnn__':{'in_axes':(0,0, 0, 0, 0, 0,0,0, None),
                         'out_axes':0,
                         'variable_axes':variable_axes_args,
                         'split_rngs':split_rngs_args,
                         'axis_name':'batch'},
            '__call_ar__':{'in_axes':(0, 0, 0, 0),
                         'out_axes':0,
                         'variable_axes':variable_axes_args,
                         'split_rngs':split_rngs_args,
                         'axis_name':'batch'}})


# ---------------------------------------------------------------------------
# Hierarchical no-book model (ported from exp_R11_book-ablation)
# ---------------------------------------------------------------------------

class HierarchicalLobPredModel(nn.Module):
    """Message-only model with hierarchical 2-stage architecture.

    Mirrors PaddedLobPredModel's structure (message_encoder + Dense bottleneck + fused_s5)
    but without the book encoder. The Dense bottleneck is (d_model -> d_model) instead of
    (2*d_model -> d_model), so there's no concatenation with book features.

    This tests whether removing the book encoder is viable at longer context lengths,
    where the message history alone should be sufficient to reconstruct book state.
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int  # fused layers
    n_message_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    mode: str = "none"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        self.message_encoder = StackedEncoderModel(
            ssm=self.ssm, d_model=self.d_model,
            n_layers=self.n_message_layers,
            activation=self.activation, dropout=self.dropout,
            training=self.training, prenorm=self.prenorm,
            batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=True, vocab_size=self.d_output,
        )
        # fused_s5 with use_embed_layer=False creates Dense(d_model -> d_model) bottleneck
        self.fused_s5 = StackedEncoderModel(
            ssm=self.ssm, d_model=self.d_model,
            n_layers=self.n_layers,
            activation=self.activation, dropout=self.dropout,
            training=self.training, prenorm=self.prenorm,
            batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=False,
        )
        self.decoder = nn.Dense(self.d_output)

    def __call_ar__(self, x, integration_timesteps):
        x = self.message_encoder(x, integration_timesteps)
        x = self.fused_s5(x, jnp.ones(x.shape[0]))
        x = self.decoder(x)
        return nn.log_softmax(x.astype(jnp.float32), axis=-1)

    def __call_rnn__(self, hidden, x, d, integration_timesteps):
        hiddens_m, hiddens_f, ema = hidden
        hiddens_m, x = self.message_encoder.__call_rnn__(hiddens_m, x, d, integration_timesteps)
        hiddens_f, x = self.fused_s5.__call_rnn__(hiddens_f, x, d, jnp.ones(x.shape[0]))
        fo, override = ema
        x = self.decoder(x)
        return (hiddens_m, hiddens_f, (fo, jnp.zeros_like(override))), nn.log_softmax(x.astype(jnp.float32), axis=-1)

    @staticmethod
    def initialize_carry(batch_size, hidden_size,
                         n_message_layers=2, n_fused_layers=6,
                         h_size_ema=256, **gdn_kwargs):
        return (
            StackedEncoderModel.initialize_carry(
                batch_size, hidden_size, n_message_layers, **gdn_kwargs),
            StackedEncoderModel.initialize_carry(
                batch_size, hidden_size, n_fused_layers, **gdn_kwargs),
            (jnp.zeros((batch_size, 1, h_size_ema)),
             jnp.ones((batch_size, 1, 1))),
        )


BatchHierarchicalLobPredModel = nn.vmap(
    HierarchicalLobPredModel,
    in_axes=(0, 0), out_axes=0,
    variable_axes=variable_axes_args,
    split_rngs=split_rngs_args, axis_name='batch',
    methods={
        '__call_ar__': {'in_axes': (0, 0), 'out_axes': 0,
                        'variable_axes': variable_axes_args,
                        'split_rngs': split_rngs_args,
                        'axis_name': 'batch'},
        '__call_rnn__': {'in_axes': (0, 0, 0, 0), 'out_axes': 0,
                         'variable_axes': variable_axes_args,
                         'split_rngs': split_rngs_args,
                         'axis_name': 'batch'},
    })


class HierarchicalNoBookInferenceWrapper(nn.Module):
    """Wraps HierarchicalLobPredModel with PaddedLobPredModel's 4/8-arg interface.

    Accepts book inputs (required by inference pipeline) but discards them.
    Param tree: {message_encoder, fused_s5, decoder} matching HierarchicalLobPredModel.
    """
    ssm: nn.Module
    d_output: int
    d_model: int
    n_layers: int
    n_message_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.0
    training: bool = True
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        self.message_encoder = StackedEncoderModel(
            ssm=self.ssm, d_model=self.d_model,
            n_layers=self.n_message_layers,
            activation=self.activation, dropout=self.dropout,
            training=self.training, prenorm=self.prenorm,
            batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=True, vocab_size=self.d_output,
        )
        self.fused_s5 = StackedEncoderModel(
            ssm=self.ssm, d_model=self.d_model,
            n_layers=self.n_layers,
            activation=self.activation, dropout=self.dropout,
            training=self.training, prenorm=self.prenorm,
            batchnorm=self.batchnorm, bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
            use_embed_layer=False,
        )
        self.decoder = nn.Dense(self.d_output)

    def __call__(self, x_m, x_b, msg_ts, book_ts):
        x = self.message_encoder(x_m, msg_ts)
        x = self.fused_s5(x, jnp.ones(x.shape[0]))
        x = self.decoder(x)
        return nn.log_softmax(x.astype(jnp.float32), axis=-1)

    def __call_ar__(self, x_m, x_b, msg_ts, book_ts):
        x = self.message_encoder(x_m, msg_ts)
        x = self.fused_s5(x, jnp.ones(x.shape[0]))
        x = self.decoder(x)
        return nn.log_softmax(x.astype(jnp.float32), axis=-1)

    def __call_rnn__(self, hiddens_tuple, x_m, x_b, d_m, d_b, d_f, msg_ts, book_ts):
        hiddens_m, _, hiddens_f, ema = hiddens_tuple
        hiddens_m, x = self.message_encoder.__call_rnn__(hiddens_m, x_m, d_m, msg_ts)
        hiddens_f, x = self.fused_s5.__call_rnn__(hiddens_f, x, d_f, jnp.ones(x.shape[0]))
        x = self.decoder(x)
        logits = nn.log_softmax(x.astype(jnp.float32), axis=-1)
        return (hiddens_m, hiddens_tuple[1], hiddens_f, ema), logits

    @staticmethod
    def initialize_carry(batch_size, hidden_size,
                         n_message_layers=2,
                         n_book_pre_layers=0, n_book_post_layers=0,
                         n_fused_layers=6,
                         h_size_ema=256, **gdn_kwargs):
        return (
            StackedEncoderModel.initialize_carry(
                batch_size, hidden_size, n_message_layers, **gdn_kwargs),
            ([], []),  # dummy book encoder hiddens
            StackedEncoderModel.initialize_carry(
                batch_size, hidden_size, n_fused_layers, **gdn_kwargs),
            (jnp.zeros((batch_size, 1, h_size_ema)),
             jnp.ones((batch_size, 1, 1))),
        )


BatchHierarchicalNoBookInferenceWrapper = nn.vmap(
    HierarchicalNoBookInferenceWrapper,
    in_axes=(0, 0, 0, 0), out_axes=0,
    variable_axes=variable_axes_args,
    split_rngs=split_rngs_args, axis_name='batch',
    methods={
        '__call__':     {'in_axes': (0, 0, 0, 0), 'out_axes': 0,
                         'variable_axes': variable_axes_args,
                         'split_rngs': split_rngs_args,
                         'axis_name': 'batch'},
        '__call_ar__':  {'in_axes': (0, 0, 0, 0), 'out_axes': 0,
                         'variable_axes': variable_axes_args,
                         'split_rngs': split_rngs_args,
                         'axis_name': 'batch'},
        '__call_rnn__': {'in_axes': (0, 0, 0, 0, 0, 0, 0, 0), 'out_axes': 0,
                         'variable_axes': variable_axes_args,
                         'split_rngs': split_rngs_args,
                         'axis_name': 'batch'},
    })


# @partial(
#     jax.vmap,
#     in_axes=(0, None),
# )
def running_mean(x):
    return jnp.cumsum(x,axis=1)




@partial(jax.vmap,
         in_axes=(1,None),
         out_axes=(1))
def numpy_ewma_v2(data, window):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]
    pows = alpha_rev**(jnp.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)
    # jax.debug.print("pows \n {}\n offset \n {}\n pw0 \n {}\n",pows,offset,scale_arr)
    mult = pw0*scale_arr*data
    cumsums = jnp.cumsum(mult)
    out = offset + cumsums*scale_arr[::-1]
    return out


@partial(jax.jit, static_argnums=(1,))
@partial(jax.vmap,in_axes=(1,None,1,None),out_axes=(1,1))
def ewma_vectorized_safe(data, alpha, first_offset,override_fo, row_size=1000, dtype=None, order='C'):
    """
    Reshapes data before calculating EWMA, then iterates once over the rows
    to calculate the offset without precision issues
    :param data: Input data, will be flattened.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param row_size: int, optional
        The row size to use in the computation. High row sizes need higher precision,
        low values will impact performance. The optimal value depends on the
        platform and the alpha being used. Higher alpha values require lower
        row size. Default depends on dtype.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    :return: The flattened result.
    """
    data = jnp.array(data, copy=False)

    if dtype is None:
        if data.dtype == jnp.float32:
            dtype = jnp.float32
        else:
            dtype = jnp.float
    else:
        dtype = jnp.dtype(dtype)

    row_size = int(row_size) if row_size is not None else get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        out=ewma_vectorized(data, alpha,offset=first_offset, dtype=dtype, order=order)
        # The normal function can handle this input, use that
        return (out,out[-1:])

    if data.ndim > 1:
        # flatten input
        data = jnp.reshape(data, -1, order=order)

    out = jnp.empty_like(data, dtype=dtype)


    row_n = int(data.size // row_size)  # the number of rows to use
    # jax.debug.print("Row n {}",row_n)
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    # jax.debug.print("Trailing n {}",trailing_n)

    first_offset=jnp.where(override_fo,data[0],first_offset[0])

    # if override_fo:
    #     first_offset = data[0]
    # else:
    #     first_offset=first_offset[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = jnp.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = jnp.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = jnp.reshape(out, (row_n, row_size))
        data_main_view = jnp.reshape(data, (row_n, row_size))

    # jax.debug.print("outmain view {}",out_main_view)
    # jax.debug.print("daata main view {}",data_main_view)


    # get all the scaled cumulative sums with 0 offset
    out_main_view=ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                       order='C')
    
    # jax.debug.print("outmain view after ewma {}",out_main_view)


    scaling_factors = (1 - alpha) ** jnp.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # jax.debug.print("Scaling factors {}",scaling_factors)

    # create offset array
    offsets = jnp.empty(out_main_view.shape[0], dtype=dtype)

    offsets=offsets.at[0].set(first_offset)
    # iteratively calculate offset for each row

    # jax.debug.print("offsets: {}", offsets)


    for i in range(1, out_main_view.shape[0]):
        offsets=offsets.at[i].set(offsets[i - 1] * last_scaling_factor + out_main_view[i - 1,-1])

    # add the offsets to the result
    # jax.debug.print("Shapes: offset {} scaling factor {} out {}",offsets,scaling_factors,out_main_view)
    out_main_view += offsets[:, jnp.newaxis] * scaling_factors[jnp.newaxis, :]
    out=jnp.ravel(out_main_view)
    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        out_trail=ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                        dtype=dtype, order='C')
        out=jnp.concatenate([out,out_trail])
    return (out,out[-1:])

def get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    # This will return the maximum row size possible on 
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use jnp.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = jnp.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(jnp.log(epsilon)/jnp.log(1-alpha)) + 1

def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    print("CALL")
    data = jnp.array(data, copy=False)

    if dtype is None:
        if data.dtype == jnp.float32:
            dtype = jnp.float32
        else:
            dtype = jnp.float64
    else:
        dtype = jnp.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order=order)

    if out is None:
        out = jnp.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = jnp.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = jnp.power(1. - alpha, jnp.arange(data.size + 1, dtype=dtype))
    # # jax.debug.print("Scaling factors {}",scaling_factors)
    # create cumulative sum array
    out=jnp.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1])
    out=jnp.cumsum(out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    offset = jnp.array(offset, copy=False).astype(dtype, copy=False)
    # add offsets
    out += offset * scaling_factors[1:]

    return out

def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C'):
    """
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = jnp.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == jnp.float32:
            dtype = jnp.float32
        else:
            dtype = jnp.float64
    else:
        dtype = jnp.dtype(dtype)


    if data.size < 1:
        # empty input, return empty array
        return jnp.array([])

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, jnp.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T

    if offset is None:
        # use the first element of each row as the offset
        offset = jnp.copy(data[:, 0])
    elif jnp.size(offset) == 1:
        offset = jnp.reshape(offset, (1,))

    alpha = jnp.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = jnp.power(1. - alpha, jnp.arange(row_size + 1, dtype=dtype))
    # create a scaled cumulative sum array
    out=jnp.multiply(
        data,
        jnp.multiply(alpha * scaling_factors[-2], jnp.ones((row_n, 1), dtype=dtype))
        / scaling_factors[jnp.newaxis, :-1]
    )
    out=jnp.cumsum(out, axis=1)
    out /= scaling_factors[jnp.newaxis, -2::-1]

    offset = offset.astype(dtype, copy=False)
    # add the offsets to the scaled cumulative sums
    out += offset[:, jnp.newaxis] * scaling_factors[jnp.newaxis, 1:]

    return out


# ──────────────────────────────────────────────────────────────────────
# 1-token-per-message model classes
# ──────────────────────────────────────────────────────────────────────

class FieldEmbedding(nn.Module):
    """Sum of 24 per-field embeddings.

    Each token position has its own embedding table of size (V_i + 4, d_model).
    Input: (L, 24) int32 local indices
    Output: (L, d_model) float32
    """
    field_vocab_sizes: Tuple[int, ...]  # FIELD_VOCAB_SIZES_WITH_SPECIAL
    d_model: int

    @nn.compact
    def __call__(self, x):
        # x: (L, 24) int32
        out = jnp.zeros((x.shape[0], self.d_model))
        for i, v_size in enumerate(self.field_vocab_sizes):
            out = out + nn.Embed(v_size, self.d_model, name=f'field_{i}')(x[:, i])
        return out


class MultiFieldDecoder(nn.Module):
    """24 independent classification heads, one per token field.

    Input: (L, d_model) float32
    Output: list of 24 (L, V_i) log-softmax arrays
    """
    field_vocab_sizes: Tuple[int, ...]  # FIELD_VOCAB_SIZES_WITH_SPECIAL

    @nn.compact
    def __call__(self, x):
        field_logits = []
        for i, v_size in enumerate(self.field_vocab_sizes):
            logits = nn.Dense(v_size, name=f'head_{i}')(x)
            logits = nn.log_softmax(logits.astype(jnp.float32), axis=-1)
            field_logits.append(logits)
        return field_logits


class OneTokenPaddedLobPredModel(nn.Module):
    """1-token-per-message S5 model.

    Per-field embedding sum → book encoder → fused S5 → per-field decode.
    No message_encoder (embedding IS the message representation).
    """
    ssm: nn.Module
    field_vocab_sizes: Tuple[int, ...] = FIELD_VOCAB_SIZES_WITH_SPECIAL
    d_model: int = 64
    d_book: int = 40
    n_fused_layers: int = 6
    n_book_pre_layers: int = 1
    n_book_post_layers: int = 1
    activation: str = "gelu"
    dropout: float = 0.2
    training: bool = True
    mode: str = "pool"
    prenorm: bool = False
    batchnorm: bool = False
    bn_momentum: float = 0.9
    step_rescale: float = 1.0

    def setup(self):
        self.field_embedding = FieldEmbedding(
            field_vocab_sizes=self.field_vocab_sizes,
            d_model=self.d_model,
        )

        self.book_encoder = LobBookModel(
            ssm=self.ssm,
            d_book=self.d_book,
            d_model=self.d_model,
            n_pre_layers=self.n_book_pre_layers,
            n_post_layers=self.n_book_post_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )

        self.fused_s5 = StackedEncoderModel(
            ssm=self.ssm,
            d_model=self.d_model,
            n_layers=self.n_fused_layers,
            activation=self.activation,
            dropout=self.dropout,
            training=self.training,
            prenorm=self.prenorm,
            batchnorm=self.batchnorm,
            bn_momentum=self.bn_momentum,
            step_rescale=self.step_rescale,
        )

        self.decoder = MultiFieldDecoder(
            field_vocab_sizes=self.field_vocab_sizes,
        )

    def __call__(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps):
        x_m = self.field_embedding(x_m)
        x_b = self.book_encoder(x_b, book_integration_timesteps)
        x = jnp.concatenate([x_m, x_b], axis=1)
        x = self.fused_s5(x, jnp.ones(x.shape[0]))

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        elif self.mode in ["none"]:
            pass
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last', 'none']")

        return self.decoder(x)

    def __call_rnn__(self, hiddens_tuple,
                      x_m, x_b,
                      d_b, d_f,
                      message_integration_timesteps, book_integration_timesteps):
        """RNN-mode forward for step-by-step inference.

        Args:
            hiddens_tuple: 3-tuple (hiddens_b, hiddens_fused, ema)
            x_m: message input (L, 24) local field indices
            x_b: book state (L_b, book_dim)
            d_b, d_f: done flags for book and fused encoders
            message_integration_timesteps, book_integration_timesteps: timesteps
        Returns:
            (new_hiddens_tuple, field_logits) where field_logits is list of 24 arrays
        """
        hiddens_b, hiddens_fused, ema = hiddens_tuple
        fo, override = ema

        # FieldEmbedding is algebraic — no hidden state
        x_m = self.field_embedding(x_m)            # (L, 24) → (L, d_model)

        # Book encoder RNN step
        hiddens_b, x_b = self.book_encoder.__call_rnn__(hiddens_b, x_b, d_b, book_integration_timesteps)

        x = jnp.concatenate([x_m, x_b], axis=1)
        hiddens_fused, x = self.fused_s5.__call_rnn__(hiddens_fused, x, d_f, jnp.ones(x.shape[0]))

        if self.mode in ["pool"]:
            x = jnp.mean(x, axis=0)
        elif self.mode in ["last"]:
            x = x[-1]
        elif self.mode in ["none"]:
            pass
        elif self.mode in ['ema']:
            x, fo = ewma_vectorized_safe(x, 2 / (22 + 1.0), fo, override)
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last', 'none', 'ema']")

        field_logits = self.decoder(x)  # list of 24 log-softmax arrays
        return (hiddens_b, hiddens_fused, (fo, jnp.zeros_like(override))), field_logits

    def __call_ar__(self, x_m, x_b, message_integration_timesteps, book_integration_timesteps):
        """Autoregressive forward: full sequence output, no pooling."""
        x_m = self.field_embedding(x_m)
        x_b = self.book_encoder(x_b, book_integration_timesteps)
        x = jnp.concatenate([x_m, x_b], axis=1)
        x = self.fused_s5(x, jnp.ones(x.shape[0]))
        return self.decoder(x)

    @staticmethod
    def initialize_carry(batch_size, hidden_size,
                         n_book_pre_layers=1, n_book_post_layers=1,
                         n_fused_layers=6, h_size_ema=512,
                         **kwargs):
        """Initialize hidden state carry for 1tok model (no message_encoder)."""
        # 3-tuple: book, fused, ema (no message_encoder carry)
        h_tuple_init = (
            LobBookModel.initialize_carry(
                batch_size, hidden_size, n_book_pre_layers, n_book_post_layers, **kwargs),
            StackedEncoderModel.initialize_carry(
                batch_size, hidden_size, n_fused_layers, **kwargs),
            (jnp.zeros((batch_size, 1, h_size_ema)),
             jnp.ones((batch_size, 1, 1))),
        )
        return h_tuple_init


BatchOneTokenPaddedLobPredModel = nn.vmap(
    OneTokenPaddedLobPredModel,
    in_axes=(0, 0, 0, 0),
    out_axes=0,
    variable_axes=variable_axes_args,
    split_rngs=split_rngs_args, axis_name='batch',
    methods={
        '__call__': {
            'in_axes': (0, 0, 0, 0),
            'out_axes': 0,
            'variable_axes': variable_axes_args,
            'split_rngs': split_rngs_args,
            'axis_name': 'batch',
        },
        '__call_rnn__': {
            'in_axes': (0, 0, 0, 0, 0, 0, 0),  # hiddens, x_m, x_b, d_b, d_f, msg_ts, book_ts
            'out_axes': 0,
            'variable_axes': variable_axes_args,
            'split_rngs': split_rngs_args,
            'axis_name': 'batch',
        },
        '__call_ar__': {
            'in_axes': (0, 0, 0, 0),
            'out_axes': 0,
            'variable_axes': variable_axes_args,
            'split_rngs': split_rngs_args,
            'axis_name': 'batch',
        },
    })