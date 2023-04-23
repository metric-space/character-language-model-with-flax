import os
import re
import requests
import hashlib
import collections
import sys

import flax.linen as nn
from flax.linen import initializers
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import struct
from flax.core.frozen_dict import freeze, unfreeze

from typing import Any, Callable, Iterable, Optional, Tuple, Union, Dict

"""

TODOs
1. get the dataset stuff working
2. figure out how to get MLP with the RNN working
3. Figure out the loss stuff
4. Figure out the training loop
5. Figure out the validation loop
6. graph the results

"""

Array = jnp.ndarray
TrainState = train_state.TrainState


class Metrics(struct.PyTreeNode):
  """Computed metrics."""
  loss: float
  accuracy: float
  count: Optional[int] = None

SEED = 421
rng = jax.random.PRNGKey(SEED)

DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.
    """

    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

class Vocab:
    """Output : Map Text Int  """
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        # TODO: figure out why this doesn't work
        #if hasattr(indices, '__len__') and len(indices) > 1:
        #    return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']


class TimeMachine():
    """
    The Time Machine dataset.
    """
    def _download(self):
        fname = download(DATA_URL + 'timemachine.txt', "./DATA", '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        """ :: Text -> Text  """
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        """ :: Text -> [Text] """
        return list(text)

    # TODO: Vocab 
    def build(self, raw_text, vocab=None):
        """Defined in :numref:`sec_text-sequence`"""
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        """Defined in :numref:`sec_language-model`"""
        corpus, self.vocab = self.build(self._download())
        # array :: [[Int]]
        array = jnp.array([corpus[i:i+num_steps+1]
                            for i in range(len(corpus)-num_steps)])
        # shifting arrays
        self.X, self.Y = array[:,:-1], array[:,1:]

    def data(self):
        return self.X, self.Y

    def __len__(self):
        # return unique number of words
        return len(self.vocab)


class RNN(nn.Module):
    num_inputs: int
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, state=None):

        W_xh = self.param('W_xh', initializers.lecun_normal(), [self.num_inputs, self.num_hiddens])
        W_hh = self.param('W_hh', initializers.lecun_normal(), [self.num_hiddens, self.num_hiddens])
        b_h = self.param('b_h', initializers.zeros_init(), [self.num_hiddens])

        outputs = []
        for X in inputs: # Shape of inputs: (num_steps, batch_size,  num_inputs)
            state = jnp.tanh(jnp.matmul(X, W_xh) + (jnp.matmul(state, W_hh) if state is not None else 0) + b_h)
            outputs.append(state)
        return jnp.array(outputs), state 


def one_hot(X, n_class, dtype=jnp.float32):
    return jax.nn.one_hot(X, n_class).astype(dtype)

class Combo(nn.Module):
    feature_length: int
    rnn: nn.Module

    def setup(self):
        self.dense = nn.Dense(self.feature_length)

    def call_rnn(self, inputs, state=None):
        return self.rnn(inputs, state)

    def call_dense(self, inputs):
        return self.dense(inputs)

    def __call__(self, inputs, state=None):
        (outputs, state) = self.rnn(inputs, state)
        outputs = self.dense(outputs)
        return outputs, state


def predict(nn, params, num_predict, vocab, prefix):

    outputs = [vocab[prefix[0]]]

    state = None

    for t in range(num_predict + len(prefix) - 1):
        X = one_hot(outputs[-1], len(vocab))
        ans = nn.apply(params, jnp.array([[X]]), state=state, method='call_rnn')

        (Y, state) = ans
        if t < len(prefix) - 1:
            outputs.append(vocab[prefix[t + 1]])
        else:
            x = nn.apply(params, Y, method='call_dense')
            outputs.append(int(x.argmax(axis=2)))
    print(outputs)

    return ''.join([vocab.idx_to_token[i] for i in outputs])


# ==================================================


def create_train_state(rng, model , params):
  """Create initial training state."""
  tx = optax.chain(
      optax.sgd(learning_rate=3e-2, momentum=0.9),
      # optax.additive_weight_decay(0.2),
      optax.clip(3.0),
      )
  state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
  return state


def compute_metrics(labels: Array, logits: Array) -> Metrics:
  """Computes the metrics, summed across the batch if a batch is provided."""
  loss = optax.softmax_cross_entropy_with_integer_labels(labels=labels, logits=logits)
  predictions = logits.argmax(axis=-1)
  accuracy = jnp.sum(predictions == labels)/labels.shape[0]
  return Metrics(
      loss=jnp.sum(loss),
      accuracy=accuracy,
      count=logits.shape[0])


def train_step(
    state: TrainState,
    X_: Array,
    Y_: Array,
    carry: jnp.ndarray,
    rngs: Dict[str, Any],
) -> (TrainState, Metrics):
  """Train for a single step."""
  # Make sure to get a new RNG at every step.
  step = state.step
  rngs = {name: jax.random.fold_in(rng, step) for name, rng in rngs.items()}

  def loss_fn(params):
    variables = params
    logits = state.apply_fn( variables, X_, carry, rngs=rngs)

    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=jnp.array(logits[0]), labels=Y_))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  # output is of grad_fn is ((value, auxiliary_data), gradient)
  value, grads = grad_fn(state.params)
  _, aux_data = value

  logits, new_carry = aux_data

  new_state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(labels=Y_, logits=jnp.array(logits))
  return new_state, new_carry, metrics


# =============== Data bit =========================

num_steps    = 25
time_machine = TimeMachine(None, num_steps)
X,Y = time_machine.data()

batch_size   = 1
vocab_length = len(time_machine) # this should be vocab size
num_hiddens  = 100 

rng, key1, key2, key3 = jax.random.split(rng, 4)

rnn = RNN(vocab_length, num_hiddens)

def print_carry_shape(carry):
    print("carry shape is ", carry[0].shape, carry[1].shape)

# carry = nn.LSTMCell.initialize_carry(key1, (num_hiddens,), vocab_length)
combo = Combo(vocab_length, rnn)
input_data = jax.lax.map(lambda y:jax.nn.one_hot(y.T, vocab_length), X[0])
combo_params = combo.init(key2, jnp.expand_dims(input_data, 1))
print(combo.tabulate(key2, jnp.expand_dims(input_data, 1)))
state = create_train_state(key3, combo, combo_params)

carry = None

for i, x in enumerate(X):
    if i % 100 == 0 and carry != None:
        print("> ", predict(combo, state.params, 40, time_machine.vocab, [time_machine.vocab.to_tokens(x[0])]))
    x_ = jax.lax.map(lambda y:jax.nn.one_hot(y, vocab_length), x)
    x_ = jnp.expand_dims(x_, 1)
    y_ = jnp.expand_dims(Y[i],-1)
    # y_ = jax.lax.map(lambda x:jax.nn.one_hot(x.T, vocab_length, dtype=jnp.int32), Y[i])
    # y_ = jnp.expand_dims(y_, 0)
    rng, epoch_rng = jax.random.split(rng)
    rngs = {'dropout': epoch_rng}
    state, carry, train_metrics = train_step(state, x_, y_, carry, rngs)
    print(train_metrics)
