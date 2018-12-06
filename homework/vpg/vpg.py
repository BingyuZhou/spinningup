import tensorflow as tf
import numpy as np
import gym
import tqdm
import argparse
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


class vpg_buffer:
    """
    Replay Buffer (On policy)
    - Only contain the transitions from the current policy
    - overwritten when a new policy is applied
    """
    def __init__(self, buffer_size, obs_dim, action_dim, gamma, lamb):
        self.buffer_size = buffer_size
        self.dim =dim
        self.state = np.zeros(shape=(buffer_size, obs_dim))
        self.action = np.zeros(shape=(buffer_size, action_dim))
        self.reward = np.zeros(shape=(buffer_size,1))
        self.value = np.zeros(shape=(buffer_size,1))
        self.logp = np.zeros(shape=(buffer_size,1))
        self.gamma =gamma
        self.lamb = lamb
    
    def add(self, s, a, r, v, logp, step):
        self.state[step] = s
        self.action[step]=a
        self.reward[step]=r
        self.value[step]=v
        self.logp[step]=logp
    
    def final(self, v,index):
        """
        If the game is terminated, v = 0
        Else we use the value from the value function to bootstrap the rewards to go
        """
        self.reward[index] = v
        self.value[index]=v

        # Compute rewards to go 
        gamma_mat = np.zeros((index+1, index+1))
        base = 1
        np.fill_diagonal(gamma_mat, 1)
        for i in range(1,index+1):
            base = self.gamma*base
            np.fill_diagonal(gamma_mat[:,i], base)
        
        self.rewards_to_go = np.dot(gamma_mat, self.reward[:index+1])
        self.rewards_to_go =self.rewards_to_go[:-1]

        # Compute advantage (GAE)
        # A = sum (gamma * lambda)^t * TD
        td_vec = self.reward[:index]+self.gamma*self.value[1:]-self.value[:index]

        gamma_lamb_matrix = np.zeros((index, index))
        base =1
        np.fill_diagonal(gamma_lamb_matrix, 1)
        for i in range(1, index):
            base = self.gamma*self.lamb*base
            np.fill_diagonal(gamma_lamb_matrix[:,i], base)

        self.advantage = np.dot(gamma_lamb_matrix, td_vec)
    
    def sample(self, size):
        









def vpg(env, actor_critic_fn, epoch, episode, steps_per_episode, pi_lr, v_lr,
        gamma, lamb, hid):
    """
    Vanilla policy gradeint
    with Generalized Advantage Estimation (GAE)
    - On poliocy
    - suitable for discrete and continous action space
    """
    buffer = vpg_buffer()

    act_dim = env.action_space.shape
    obs_dim = env.observation_space.shape

    if isinstance(env.action_space, gym.spaces.Box):
        policy_samp = 'diagnoal_gaussian'
    elif isinstance(env.action_space, gym.spaces.Discrete):
        policy_samp = 'categorical'
    s = tf.placeholder(dtype=tf.float32, shape=obs_dim, name='obs')
    a = tf.placeholder(dtype=tf.float32, shape=act_dim, name='action')
    adv = tf.placeholder(dtype=tf.float32, shape=None, name='advantage')
    r_to_go = tf.placeholder(
        dtype=tf.float32, shape=None, name='rewards_to_go')

    pi, logp, logp_pi, v = actor_critic_fn(s, a, act_dim, hid, hid,
                                           policy_samp)
    pi_loss = -tf.reduce_sum(logp_pi * adv)
    v_loss = tf.reduce_sum(v * r_to_go)

    pi_opt = tf.train.AdamOptimizer(pi_lr).minimize(pi_loss)
    v_opt = tf.train.AdamOptimizer(v_lr).minimize(v_loss)

    with tf.Session() as sess:
        for ep in epoch:
            for es in episode:
                ob = env.reset()  # initial state
                r_t = 0

                for step in steps_per_episode:
                    a_t, v_t, logp_t = sess.run(
                        [pi, v, logp_pi], feed_dict={s: ob})
                    buffer.add(ob, a_t, r_t, v_t, logp_t,step)
                    ob, r, done, info = env.step(a_t)

                    if done:
                        buffer.final(v=0, index=step+1)
                        break
                    elif (step == steps_per_episode - 1):
                        buffer.final(v=v, index=step)
            
            # Update policy 
            sess.run(pi_opt, feed_dict={s:,a:,adv:})
            
            # Update value function
            sess.run(v_opt,feed_dict={r_to_go:})




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for vpg')
    parser.add_argument('--env', type=str, default='CartPole-v2')
    parser.add_argument('--pi_lr', type=float, default=0.01)
    parser.add_argument('--v_lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--episode', type=int, default=20)
    parser.add_argument('--hid', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--lamb', type=float, default=0.93)
    args = parser.parse_args()

    env = gym.make(args.env)

    vpg(env, actor_critic, args.epoch, args.episode, args.steps_per_episode,
        args.pi_lr, args.v_lr, args.gamma, args.lamb, args.hid)
