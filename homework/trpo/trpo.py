import tensorflow as tf
import numpy as np
import gym
import tqdm
import argparse
import scipy.signal
import time
from spinup.utils.logx import EpochLogger

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

    def kl(self, logp, logq):
        return tf.reduce_sum(tf.exp(logp)*(logp-logq), axis=1)

    def run(self, s, a):
        logits = mlp(
            s,
            self._hid,
            self._action_dim,
            activation=tf.nn.tanh,
            output_activation=None)
        logp_all = tf.nn.log_softmax(logits)  # batch_size x action_dim, [n ,m]
        # batch_size x action_index, [n, 1]
        pi = tf.squeeze(tf.multinomial(logits, 1))
        logp_pi = tf.reduce_sum(
            tf.one_hot(pi, depth=self._action_dim) * logp_all,
            axis=1)  # log probability of policy action at current state
        logp = tf.reduce_sum(
            tf.one_hot(a, depth=self._action_dim) * logp_all,
            axis=1)  # log probability of action a at current state
        pi_dist_old = tf.placeholder(dtype=tf.float32, shape=(None,))
        d_kl = self.kl(logp_all, pi_dist_old)
        return pi, logp, logp_pi, logp_all, d_kl


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

    def kl(self, mu_0, logstd_0, p_1):
        """ KL divergence 0 over 1"""
        var_0 = tf.exp(logstd_0)**2
        mu_1 = p_1[:, 0]
        logstd_1 = p_1[:, -1]
        var_1 = tf.exp(logstd_1)**2

        return tf.reduce_sum((0.5*((var_0 + (mu_1 - mu_0)**2)/(var_1+EPS) - 1)) + logstd_0 - logstd_1)

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
        pi_dist_old = tf.placeholder(dtype=tf.float32, shape=(None, 2))
        d_kl = self.kl(mu, std, pi_dist_old)
        return pi, logp, logp_pi, [mu, self.logstd], d_kl


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

        pi, logp, logp_pi, pi_dist, d_kl = actor.run(s, a)

    with tf.variable_scope('v'):
        v = mlp(s, value_hid, 1, activation=tf.nn.tanh, output_activation=None)

    return pi, logp, logp_pi, pi_dist, d_kl, v


class trpo_buffer:
    """
    Replay Buffer (On policy)
    - Only contain the transitions from the current policy
    - overwritten when a new policy is applied
    """

    def __init__(self, buffer_size, obs_dim, action_dim, gamma, lamb, policy_sample):
        self.buffer_size = buffer_size
        self.state_buffer = np.zeros(shape=(buffer_size, *obs_dim))
        self.action_buffer = np.zeros(shape=(buffer_size, *action_dim))
        self.reward_buffer = np.zeros(shape=(buffer_size, ))
        self.value_buffer = np.zeros(shape=(buffer_size, ))
        self.logp_buffer = np.zeros(shape=(buffer_size, ))
        self.advantage_buffer = np.zeros(shape=(buffer_size, ))
        self.rewards_to_go_buffer = np.zeros(shape=(buffer_size, ))
        # policy distribution used for KL
        if policy_sample == 'categorical':
            self.pidist_buffer = np.zeros(shape=(buffer_size,))
        else:
            self.pidist_buffer = np.zeros(shape=(buffer_size, 2))  # mu, logstd
        self.gamma = gamma
        self.lamb = lamb
        self.path_start_index = 0

    def add(self, s, a, r, v, logp, pi_dist, step):
        assert step < self.buffer_size
        self.state_buffer[step] = s
        self.action_buffer[step] = a
        self.reward_buffer[step] = r
        self.value_buffer[step] = v
        self.logp_buffer[step] = logp
        self.pidist_buffer[step] = pi_dist
        self.end_index = step

    def cum_discounted_sum(self, x, discount):
        """
        Magic formula to compute cummunated sum of discounted value (Faster)
        Input: x = [x1, x2, x3]
        Output: x1+discount*x2+discount^2*x3, x2+discount*x3, x3
        """
        return scipy.signal.lfilter(
            [1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def my_cum_discounted_sum(self, x, discount, size):
        gamma_mat = np.zeros((size, size))
        base = 1
        np.fill_diagonal(gamma_mat, 1)
        for i in range(1, size):
            base = discount * base
            np.fill_diagonal(gamma_mat[:, i:], base)
        gamma_mat[0, -1] = base * discount

        return np.dot(gamma_mat, x)

    def final(self, v, my_cum_sum=False):
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
        if my_cum_sum:
            start = time.time()
            self.rewards_to_go_buffer[self.path_start_index:self.end_index +
                                      1] = self.my_cum_discounted_sum(
                                          rewards, self.gamma,
                                          rewards.shape[0])[:-1]
            tf.logging.debug(
                'time elasped for cum_discounted_sum:{}'.format(time.time() -
                                                                start))
        else:
            start = time.time()
            self.rewards_to_go_buffer[self.path_start_index:self.end_index +
                                      1] = self.cum_discounted_sum(
                                          rewards, self.gamma)[:-1]
            tf.logging.debug(
                'time elasped for cum_discounted_sum:{}'.format(time.time() -
                                                                start))

        # Compute advantage (GAE)
        # A = sum (gamma * lambda)^t * TD
        td_vec = rewards[:-1, ] + self.gamma * values[1:, ] - values[:-1, ]

        if my_cum_sum:
            self.advantage_buffer[self.path_start_index:self.end_index +
                                  1] = self.my_cum_discounted_sum(
                                      td_vec, self.gamma * self.lamb,
                                      values.shape[0] - 1)
        else:
            self.advantage_buffer[self.path_start_index:self.end_index +
                                  1] = self.cum_discounted_sum(
                                      td_vec, self.gamma * self.lamb)
        # Start new trajectory
        self.path_start_index = self.end_index + 1

    def normalize_adv(self):
        # Normalization of GAE (Very important !!)
        mu, stdd = np.mean(self.advantage_buffer), np.std(
            self.advantage_buffer)
        self.advantage_buffer = (self.advantage_buffer - mu) / stdd

    def sample(self, size):

        # Reset
        self.path_start_index = 0

        if (size > self.end_index):
            tf.logging.warn(
                'sample size is larger or equal than the buffer size, return all buffer'
            )
            return [
                self.state_buffer[:self.end_index + 1],
                self.action_buffer[:self.end_index + 1],
                self.rewards_to_go_buffer[:self.end_index + 1],
                self.logp_buffer[:self.end_index + 1],
                self.advantage_buffer[:self.end_index + 1],
                self.pidist_buffer[:self.end_index+1]
            ]
        else:
            sample_index = np.random.choice(
                self.end_index + 1, size, replace=False)
            return [
                self.state_buffer[sample_index],
                self.action_buffer[sample_index],
                self.rewards_to_go_buffer[sample_index],
                self.logp_buffer[sample_index],
                self.advantage_buffer[sample_index],
                self.pidist_buffer[sample_index]
            ]


def trpo(seed, env, actor_critic_fn, epoch, episode, steps_per_episode, pi_lr,
         v_lr, gamma, lamb, hid, buffer_size, batch_size, pi_train_itr,
         v_train_itr, cg_itr):
    """
    Vanilla policy gradeint
    with Generalized Advantage Estimation (GAE)
    - On poliocy
    - suitable for discrete and continous action space
    """
    # model saver
    logger = EpochLogger()
    logger.save_config(locals())

    seed += 10000
    tf.set_random_seed(seed)
    np.random.seed(seed)

    act_dim = env.action_space.shape
    obs_dim = env.observation_space.shape

    s = tf.placeholder(dtype=tf.float32, shape=(None, *obs_dim), name='obs')

    adv = tf.placeholder(dtype=tf.float32, shape=None, name='advantage')
    r_to_go = tf.placeholder(
        dtype=tf.float32, shape=None, name='rewards_to_go')
    logp_old = tf.placeholder(dtype=tf.float32, shape=None)

    if isinstance(env.action_space, gym.spaces.Box):
        policy_samp = 'diagnoal_gaussian'
        a = tf.placeholder(
            dtype=tf.float32, shape=(None, *act_dim), name='action')
        pi, logp, logp_pi, pi_dist, d_kl, v = actor_critic_fn(s, a, act_dim, hid, hid,
                                                              policy_samp)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        policy_samp = 'categorical'
        a = tf.placeholder(
            dtype=tf.int32, shape=(None, *act_dim), name='action')
        pi, logp, logp_pi, pi_dist, d_kl, v = actor_critic_fn(
            s, a, env.action_space.n, hid, hid, policy_samp
        )  # In discrete space, teh last layer should be the number of possible actions

    buffer = trpo_buffer(buffer_size, obs_dim, act_dim,
                         gamma, lamb, policy_samp)

    # should use logp, since we need the probability of the specific action sampled from buffer
    pi_loss = -tf.reduce_mean(tf.exp(logp-logp_old) * adv)
    v_loss = tf.reduce_mean((v - r_to_go)**2)

    v_opt = tf.train.AdamOptimizer(v_lr).minimize(v_loss)

    approx_entropy = tf.reduce_mean(-logp)

    # Core of TRPO
    def conjugate_grad(Hx, g):
        """
        Conjugate gradient method
        Given Hx = g, return x
        start with initial guess x0 = 0
        """
        x = np.zeros_like(g)
        r = g.copy()
        p = g.copy()
        rr = np.dot(r, r)
        for _ in range(cg_itr):
            alpha = rr/np.dot(p, Hx(p))

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

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logger.setup_tf_saver(
            sess, inputs={'x': s}, outputs={
                'pi': pi,
                'v': v
            })
        for ep in range(epoch):
            es_len = 0
            es_len_prev = 0
            for es in range(episode):
                ob = env.reset()  # initial state
                r_t = 0
                es_ret = 0

                for step in range(steps_per_episode):
                    a_t, v_t, logp_t = sess.run(
                        [pi, v, logp_pi], feed_dict={s: ob.reshape(1, -1)})
                    buffer.add(ob, a_t, r_t, v_t, logp_t, es_len)
                    # tf.logging.debug('state_buffer:{}'.format(ob))
                    # tf.logging.debug('action_buffer:{}'.format(a_t))
                    # tf.logging.debug('r:{}'.format(r_t))
                    # tf.logging.debug('logp:{}'.format(logp_t))
                    # tf.logging.debug('v:{}'.format(v_t))
                    ob, r_t, done, _ = env.step(a_t)
                    es_ret += r_t
                    es_len += 1

                    if done or step == steps_per_episode - 1:
                        if done:
                            buffer.final(v=r_t)
                            logger.store(
                                EpRet=es_ret, EpLen=es_len - es_len_prev)

                        else:
                            buffer.final(
                                sess.run(v, feed_dict={s: ob.reshape(1, -1)}))
                        ob = env.reset()
                        r_t = 0
                        es_ret = 0
                        es_len_prev = es_len
            buffer.normalize_adv()
            batch_tuple_all = buffer.sample(episode * steps_per_episode)
            pi_loss_old, v_loss_old = sess.run(
                [pi_loss, v_loss],
                feed_dict={
                    s: batch_tuple_all[0],
                    a: batch_tuple_all[1],
                    adv: batch_tuple_all[4],
                    r_to_go: batch_tuple_all[2]
                })
            # Update policy
            for _ in range(pi_train_itr):
                batch_tuple = buffer.sample(batch_size)
                assert batch_tuple[0].shape[0] == batch_size
                sess.run(
                    pi_opt,
                    feed_dict={
                        s: batch_tuple[0],
                        a: batch_tuple[1],
                        adv: batch_tuple[4]
                    })

            for i in range(v_train_itr):
                # Update value function
                batch_tuple = buffer.sample(batch_size)
                sess.run(
                    v_opt,
                    feed_dict={
                        r_to_go: batch_tuple[2],
                        s: batch_tuple[0]
                    })

            pi_loss_new, v_loss_new, approx_kl_v, approx_entropy_v = sess.run(
                [pi_loss, v_loss, approx_kl, approx_entropy],
                feed_dict={
                    s: batch_tuple_all[0],
                    a: batch_tuple_all[1],
                    adv: batch_tuple_all[4],
                    r_to_go: batch_tuple_all[2],
                    logp_old: batch_tuple_all[3]
                })

            # Save model
            if (ep % 10 == 0) or (ep == epoch - 1):
                logger.save_state({'env': env})

            # Log
            logger.store(
                LossPi=pi_loss_old,
                LossV=v_loss_old,
                DeltaLossPi=pi_loss_new - pi_loss_old,
                DeltaLossV=v_loss_new - v_loss_old,
                KL=approx_kl_v,
                Entropy=approx_entropy_v)

            logger.log_tabular('Epoch', ep)
            logger.log_tabular('TotalEnvInteracts',
                               (ep + 1) * episode * steps_per_episode)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('DeltaLossPi', average_only=True)
            logger.log_tabular('DeltaLossV', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--pi_train_itr', type=int, default=5)
    parser.add_argument('--v_train_itr', type=int, default=64)
    args = parser.parse_args()

    env = gym.make(args.env)

    trpo(0, env, actor_critic, args.epoch, args.episode,
         args.steps_per_episode, args.pi_lr, args.v_lr, args.gamma, args.lamb,
         args.hid, args.buffer_size, args.batch_size, args.pi_train_itr,
         args.v_train_itr)
