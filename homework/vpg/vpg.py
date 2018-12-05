import tensorflow as tf
import numpy as np
import gym
import tqdm

EPS = 1E-8


def log_p_gaussian(x, mu, logstd):
    return tf.reduce_sum(
        -0.5 * ((x - mu)**2 /
                (tf.exp(logstd) + EPS)**2 + tf.log(2 * np.pi) + 2 * logstd))


def mlp(x, hid, output_layer, activation, output_activation):
    input = x
    for h in hid:
        input = tf.layers.dense(input, h, activation)
    return tf.layers.dense(input, output_layer, output_activation)


class mlp_with_categorical:
    """
    Categorical policy only suitable for discrete actions
    """

    def __init__(self, policy_hid, action_dim):
        self._hid = policy_hid
        self._action_dim = action_dim

    def run(self, s, a):
        logits = mlp(
            s,
            self._hid,
            self._action_dim,
            activation=tf.nn.tanh,
            output_activation=None)
        logp_all = tf.nn.softmax(logits)  # batch_size x action_dim, [n ,m]
        pi = tf.multinomial(logits, 1)  # batch_size x action_index, [n, 1]
        logp_pi = tf.tensordot(
            tf.one_hot(pi, depth=self._action_dim), logp_all,
            axes=1)  # log probability of policy action at current state
        logp = tf.tensordot(
            tf.one_hot(a, depth=self._action_dim), logp_all,
            axes=1)  # log probability of action a at current state
        return pi, logp, logp_pi


class mlp_with_diagonal_gaussian:
    """
    Diagonal Gaussian policy suitable for discrete and continous actions
    """

    def __init__(self, policy_hid, action_dim):
        self._hid = policy_hid
        self._action_dim = action_dim
        self.logstd = tf.get_variable(
            'log_std',
            initializer=-0.5 * np.ones(action_dim, dtype=tf.float32))

    def run(self, s, a):
        mu = mlp(
            s,
            self._hid,
            self._action_dim,
            activation=tf.nn.tanh,
            output_activation=None)
        std = tf.exp(self.logstd)
        pi = tf.random_normal(tf.shape(mu)) * std + mu
        logp = log_p_gaussian(a, mu, self.logstd)
        logp_pi = log_p_gaussian(pi, mu, self.logstd)
        return pi, logp, logp_pi


class mlp_with_diagonal_gaussian_on_state:
    pass


def actor_critic(s,
                 a,
                 r,
                 action_dim,
                 policy_hid=[64, 64],
                 value_hid=[64],
                 policy_samp='categorical'):
    """
    Actor Critic model:
    Inputs: observation, reward
    Outputs: action, logp_pi, logp, v
    """
    with tf.variable_scope('pi'):
        if policy_samp == 'categorical':
            actor = mlp_with_categorical(policy_hid, action_dim)
        else:
            actor = mlp_with_diagonal_gaussian(policy_hid, action_dim)

        pi, logp, logp_pi = actor.run(s, a)

    with tf.variable_scope('v'):
        v = mlp(s, value_hid, 1, activation=tf.nn.tanh, output_activation=None)

    return pi, logp, logp_pi, v
