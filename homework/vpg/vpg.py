import tensorflow as tf
import numpy as np
import gym
import tqdm
import argparse
import scipy.signal
EPS = 1E-8

tf.logging.set_verbosity(tf.logging.INFO)


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
        logp_all = tf.nn.log_softmax(logits)  # batch_size x action_dim, [n ,m]
        pi = tf.squeeze(tf.multinomial(logits,
                                       1))  # batch_size x action_index, [n, 1]
        logp_pi = tf.reduce_sum(
            tf.one_hot(pi, depth=self._action_dim) * logp_all,
            axis=1)  # log probability of policy action at current state
        logp = tf.reduce_sum(
            tf.one_hot(a, depth=self._action_dim) * logp_all,
            axis=1)  # log probability of action a at current state
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
        self.state_buffer = np.zeros(shape=(buffer_size, *obs_dim))
        self.action_buffer = np.zeros(shape=(buffer_size, *action_dim))
        self.reward_buffer = np.zeros(shape=(buffer_size, 1))
        self.value_buffer = np.zeros(shape=(buffer_size, 1))
        self.logp_buffer = np.zeros(shape=(buffer_size, 1))
        self.advantage_buffer = np.zeros(shape=(buffer_size, 1))
        self.rewards_to_go_buffer = np.zeros(shape=(buffer_size, 1))
        self.gamma = gamma
        self.lamb = lamb
        self.path_start_index = 0

    def add(self, s, a, r, v, logp, step):
        assert step < self.buffer_size
        self.state_buffer[step] = s
        self.action_buffer[step] = a
        self.reward_buffer[step] = r
        self.value_buffer[step] = v
        self.logp_buffer[step] = logp
        self.end_index = step

    def final(self, v):
        """
        If the game is terminated, v = 0
        Else we use the value from the value function to bootstrap the rewards to go
        """
        self.reward_buffer[self.end_index + 1] = v
        self.value_buffer[self.end_index + 1] = v

        # Reward and value for current trajectory
        rewards = self.reward_buffer[self.path_start_index:self.end_index + 2]
        values = self.value_buffer[self.path_start_index:self.end_index + 2]

        # Compute rewards to go
        # gamma_mat = np.zeros((self.end_index + 1 + 1, self.end_index + 1 + 1))
        # base = 1
        # np.fill_diagonal(gamma_mat, 1)
        # for i in range(1, self.end_index + 1):
        #     base = self.gamma * base
        #     np.fill_diagonal(gamma_mat[:, i:], base)
        # gamma_mat[0, -1] = base * self.gamma

        # self.rewards_to_go = np.dot(gamma_mat,
        #                             self.reward[:self.end_index + 1 + 1])
        # self.rewards_to_go = self.rewards_to_go[:-1]

        self.rewards_to_go_buffer[self.path_start_index:self.end_index +
                                  1, :] = scipy.signal.lfilter(
                                      [1], [1, float(-self.gamma)],
                                      rewards[::-1],
                                      axis=0)[::-1][:-1]

        # Compute advantage (GAE)
        # A = sum (gamma * lambda)^t * TD
        td_vec = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        # gamma_lamb_matrix = np.zeros((self.end_index + 1, self.end_index + 1))
        # base = 1
        # np.fill_diagonal(gamma_lamb_matrix, 1)
        # for i in range(1, self.end_index):
        #     base = self.gamma * self.lamb * base
        #     np.fill_diagonal(gamma_lamb_matrix[:, i:], base)
        # gamma_lamb_matrix[0, -1] = base * self.gamma * self.lamb

        # self.advantage = np.dot(gamma_lamb_matrix, td_vec)

        self.advantage_buffer[self.path_start_index:self.end_index +
                              1, :] = scipy.signal.lfilter(
                                  [1], [1, float(-self.gamma * self.lamb)],
                                  td_vec[::-1],
                                  axis=0)[::-1]

        # Start new trajectory
        self.path_start_index = self.end_index + 1

    def normalize_adv(self):
        # Normalization of GAE (Very important !!)
        mu, stdd = np.mean(self.advantage_buffer), np.std(
            self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - mu) / stdd

    def sample(self, size):
        if (size > self.end_index):
            tf.logging.warn(
                'sample size is larger or equal than the buffer size, return all buffer'
            )
            return [
                self.state_buffer[:self.end_index + 1],
                self.action_buffer[:self.end_index + 1],
                self.rewards_to_go_buffer[:self.end_index + 1],
                self.logp_buffer[:self.end_index + 1],
                self.advantage_buffer[:self.end_index + 1]
            ]
        else:
            sample_index = np.random.choice(
                self.end_index + 1, size, replace=False)
            return [
                self.state_buffer[sample_index],
                self.action_buffer[sample_index],
                self.rewards_to_go_buffer[sample_index],
                self.logp_buffer[sample_index],
                self.advantage_buffer[sample_index]
            ]


def vpg(env, actor_critic_fn, epoch, episode, steps_per_episode, pi_lr, v_lr,
        gamma, lamb, hid, buffer_size, batch_size, train_itr):
    """
    Vanilla policy gradeint
    with Generalized Advantage Estimation (GAE)
    - On poliocy
    - suitable for discrete and continous action space
    """
    act_dim = env.action_space.shape
    obs_dim = env.observation_space.shape

    buffer = vpg_buffer(buffer_size, obs_dim, act_dim, gamma, lamb)

    s = tf.placeholder(dtype=tf.float32, shape=(None, *obs_dim), name='obs')

    adv = tf.placeholder(dtype=tf.float32, shape=None, name='advantage')
    r_to_go = tf.placeholder(
        dtype=tf.float32, shape=None, name='rewards_to_go')
    if isinstance(env.action_space, gym.spaces.Box):
        policy_samp = 'diagnoal_gaussian'
        a = tf.placeholder(
            dtype=tf.float32, shape=(None, *act_dim), name='action')
        pi, logp, logp_pi, v = actor_critic_fn(s, a, act_dim, hid, hid,
                                               policy_samp)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        policy_samp = 'categorical'
        a = tf.placeholder(
            dtype=tf.int32, shape=(None, *act_dim), name='action')
        pi, logp, logp_pi, v = actor_critic_fn(
            s, a, env.action_space.n, hid, hid, policy_samp
        )  # In discrete space, teh last layer should be the number of possible actions

    pi_loss = -tf.reduce_sum(
        logp * adv
    )  # should use logp, since we need the probability of the specific action sampled from buffer
    v_loss = tf.reduce_sum((v - r_to_go)**2)

    pi_opt = tf.train.AdamOptimizer(pi_lr).minimize(pi_loss)
    v_opt = tf.train.AdamOptimizer(v_lr).minimize(v_loss)

    logp_old = tf.placeholder(dtype=tf.float32, shape=None)
    approx_kl = tf.reduce_mean(logp_old - logp)
    approx_entropy = tf.reduce_mean(-logp)

    # Number of variables
    var_pi = tf.trainable_variables(scope='pi')
    var_v = tf.trainable_variables(scope='v')
    num_pi = 0
    num_v = 0
    for v in var_pi:
        num_pi += np.prod(v.shape)
    for v in var_v:
        num_v += np.prod(v.shape)

    tf.logging.info('Number of trainable variables: pi {}, v {}'.format(
        num_pi, num_v))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for ep in range(epoch):
            es_ret_all = []
            es_len = 0
            for es in range(episode):
                ob = env.reset()  # initial state
                r_t = 0
                es_ret = 0

                for step in range(steps_per_episode):
                    a_t, v_t, logp_t = sess.run(
                        [pi, v, logp_pi], feed_dict={s: ob.reshape(1, -1)})
                    buffer.add(ob, a_t, r_t, v_t, logp_t, es_len)
                    ob, r_t, done, _ = env.step(a_t)
                    es_ret += r_t
                    es_len += 1

                    if done or step == steps_per_episode - 1:
                        if done:
                            buffer.final(v=r_t)
                            es_ret_all.append(es_ret)

                        else:
                            buffer.final(
                                sess.run(v, feed_dict={s: ob.reshape(1, -1)}))
                        ob = env.reset()
                        r_t = 0
                        es_ret = 0
            buffer.normalize_adv()
            batch_tuple = buffer.sample(batch_size)

            pi_loss_old, v_loss_old = sess.run(
                [pi_loss, v_loss],
                feed_dict={
                    s: batch_tuple[0],
                    a: batch_tuple[1],
                    adv: batch_tuple[4],
                    r_to_go: batch_tuple[2]
                })
            # Update policy
            sess.run(
                pi_opt,
                feed_dict={
                    s: batch_tuple[0],
                    a: batch_tuple[1],
                    adv: batch_tuple[4]
                })

            for i in range(train_itr):
                # Update value function
                sess.run(
                    v_opt,
                    feed_dict={
                        r_to_go: batch_tuple[2],
                        s: batch_tuple[0]
                    })

            pi_loss_new, v_loss_new, approx_kl_v, approx_entropy_v = sess.run(
                [pi_loss, v_loss, approx_kl, approx_entropy],
                feed_dict={
                    s: batch_tuple[0],
                    a: batch_tuple[1],
                    adv: batch_tuple[4],
                    r_to_go: batch_tuple[2],
                    logp_old: batch_tuple[3]
                })

            # Log
            tf.logging.info('--------------------------')
            tf.logging.info(
                '\n epoch {} \n return {} \n policy loss old {} \n policy loss new {} \n value loss old {} \n value loss new {} \n KL {} \n Entropy {} \n'.
                format(ep, np.mean(es_ret_all), pi_loss_old, pi_loss_new,
                       v_loss_old, v_loss_new, approx_kl_v, approx_entropy_v))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for vpg')
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--pi_lr', type=float, default=0.0003)
    parser.add_argument('--v_lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--episode', type=int, default=4)
    parser.add_argument('--steps_per_episode', type=int, default=1000)
    parser.add_argument('--hid', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamb', type=float, default=0.97)
    parser.add_argument('--buffer_size', type=int, default=4200)
    parser.add_argument('--batch_size', type=int, default=4000)
    parser.add_argument('--train_itr', type=int, default=80)
    args = parser.parse_args()

    env = gym.make(args.env)

    vpg(env, actor_critic, args.epoch, args.episode, args.steps_per_episode,
        args.pi_lr, args.v_lr, args.gamma, args.lamb, args.hid,
        args.buffer_size, args.batch_size, args.train_itr)
