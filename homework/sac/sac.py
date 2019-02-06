import tensorflow as tf
import numpy as np
import gym
import tqdm
import argparse
import scipy.signal
import time
from spinup.utils.logx import EpochLogger

EPS = 1e-6
LOG_STD_MAX = 2
LOG_STD_MIN = -20
tf.logging.set_verbosity(tf.logging.INFO)


def log_p_gaussian(x, mu, logstd):
    return tf.reduce_sum(
        -0.5
        * (((x - mu) / (tf.exp(logstd) + EPS)) ** 2 + np.log(2 * np.pi) + 2 * logstd),
        axis=1,
    )


def clip_but_pass_gradient(x, l=-1.0, u=1.0):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x) * clip_up + (l - x) * clip_low)


def _squash_correction(actions):
    return tf.reduce_sum(
        tf.log(clip_but_pass_gradient(1 - actions ** 2, 0, 1) + EPS), axis=1
    )


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
            output_activation=None,
        )
        logp_all = tf.nn.log_softmax(logits)  # batch_size x action_dim, [n ,m]

        mu = tf.argmax(logp_all, axis=1, output_type=tf.int32)

        # most-likely action
        pi = tf.squeeze(
            tf.multinomial(logits, 1, output_dtype=tf.int32), axis=1
        )  # batch_size x action_index, [n, 1]
        logp_pi = tf.reduce_sum(
            tf.one_hot(pi, depth=self._action_dim) * logp_all, axis=1
        )  # log probability of policy action at current state

        return mu, pi, logp_pi


class mlp_with_diagonal_gaussian:
    def __init__(self, hid, action_dim, act_ub):
        self._hid = hid
        self._action_dim = action_dim
        self._act_ub = act_ub

    def run(self, s, a):
        net = mlp(
            s,
            self._hid[:-1],
            self._hid[-1],
            activation=tf.nn.relu,
            output_activation=tf.nn.relu,
        )
        mu = tf.layers.dense(net, *self._action_dim, activation=None)
        # std dev is dependent on state, instead of a shared-across-states learnable parameters
        log_std = tf.layers.dense(net, *self._action_dim, activation=tf.nn.tanh)
        # Mapping from (-1,1) to (LOG_STD_MIN, LOG_STD_MAX)
        log_std = 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * log_std + 0.5 * (
            LOG_STD_MAX + LOG_STD_MIN
        )
        std = tf.exp(log_std)
        pi = tf.tanh(mu + tf.random_normal(tf.shape(mu)) * std)
        mu = tf.tanh(mu)
        # log of squashed likelihood, check paper [SAC](https://arxiv.org/pdf/1801.01290.pdf)
        log_pi = log_p_gaussian(pi, mu, log_std) - _squash_correction(pi)

        mu *= self._act_ub
        pi *= self._act_ub
        return mu, pi, log_pi


def actor_critic(s, a, action_space, policy_hid=[64, 64], q_hid=[64], v_hid=[64]):
    """
    Actor Critic model:
    Inputs: observation, reward
    Outputs: action, logp_pi, logp, v
    """
    with tf.variable_scope("pi"):
        if isinstance(action_space, gym.spaces.Box):
            actor = mlp_with_diagonal_gaussian(
                policy_hid, action_space.shape, action_space.high[0]
            )
        elif isinstance(action_space, gym.spaces.Discrete):
            actor = mlp_with_categorical(policy_hid, action_space.n)
        else:
            raise Exception("Unknown action space type")

        mu, pi, log_pi = actor.run(s, a)

    # Important: squeeze is very important, it makes sure the q value is with size (batch_size,)
    if a.dtype == "int32":
        a = tf.cast(tf.reshape(a, (-1, 1)), tf.float32)
    with tf.variable_scope("q1"):
        q1 = tf.squeeze(
            mlp(
                tf.concat([s, a], -1),
                q_hid,
                1,
                activation=tf.nn.relu,
                output_activation=None,
            ),
            axis=1,
        )
    with tf.variable_scope("q2"):
        q2 = tf.squeeze(
            mlp(
                tf.concat([s, a], -1),
                q_hid,
                1,
                activation=tf.nn.relu,
                output_activation=None,
            ),
            axis=1,
        )
    pi_f = tf.cast(pi, tf.float32)
    if pi.dtype == "int32":
        pi_f = tf.cast(tf.reshape(pi, (-1, 1)), tf.float32)

    with tf.variable_scope("q1", reuse=True):
        q1_pi = tf.squeeze(
            mlp(
                tf.concat([s, pi_f], -1),
                q_hid,
                1,
                activation=tf.nn.relu,
                output_activation=None,
            ),
            axis=1,
        )
    with tf.variable_scope("q2", reuse=True):
        q2_pi = tf.squeeze(
            mlp(
                tf.concat([s, pi_f], -1),
                q_hid,
                1,
                activation=tf.nn.relu,
                output_activation=None,
            ),
            axis=1,
        )
    with tf.variable_scope("v"):
        v = tf.squeeze(
            mlp(s, v_hid, 1, activation=tf.nn.relu, output_activation=None), axis=1
        )

    return mu, pi, q1, q2, q1_pi, q2_pi, v, log_pi


class sac_buffer:
    """
    Replay Buffer (Off policy)
    - Large enough to collect all history transactions
    """

    def __init__(self, buffer_size, obs_dim, action_dim, gamma):
        self.buffer_size = buffer_size
        self.state_buffer = np.zeros(shape=(buffer_size, *obs_dim))
        self.next_state_buffer = np.zeros(shape=(buffer_size, *obs_dim))
        self.action_buffer = np.zeros(shape=(buffer_size, *action_dim))
        self.reward_buffer = np.zeros(shape=(buffer_size,))
        self.termnt_buffer = np.zeros(shape=(buffer_size,))
        self.gamma = gamma
        self.step = 0
        self.size = 0

    def add(self, s, a, r, s_next, termnt):
        assert self.step < self.buffer_size
        self.state_buffer[self.step] = s
        self.action_buffer[self.step] = a
        self.reward_buffer[self.step] = r
        self.next_state_buffer[self.step] = s_next
        self.termnt_buffer[self.step] = termnt
        self.step = (self.step + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        if batch_size > self.size:
            tf.logging.debug(
                "sample size is larger or equal than the buffer size, return all buffer"
            )
            return [
                self.state_buffer[: self.size],
                self.action_buffer[: self.size],
                self.reward_buffer[: self.size],
                self.next_state_buffer[: self.size],
                self.termnt_buffer[: self.size],
            ]
        else:
            sample_index = np.random.choice(self.size, batch_size, replace=False)
            return [
                self.state_buffer[sample_index],
                self.action_buffer[sample_index],
                self.reward_buffer[sample_index],
                self.next_state_buffer[sample_index],
                self.termnt_buffer[sample_index],
            ]


def sac(
    seed,
    env_fn,
    actor_critic_fn,
    epoch,
    episode,
    steps_per_episode,
    pi_lr,
    q_lr,
    v_lr,
    gamma,
    hid,
    buffer_size,
    batch_size,
    logger_kwargs,
    rho,
    alpha,
    start_steps=1e4,
):
    """
    Soft Actor Critic
    - off policy
    - stochastic policy
    - entropy based
    - continous action
    """
    # model saver
    logger = EpochLogger(**logger_kwargs)

    logger.save_config(locals())

    seed += 10000
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    act_space = env.action_space
    act_dim = env.action_space.shape

    obs_dim = env.observation_space.shape

    s = tf.placeholder(dtype=tf.float32, shape=(None, *obs_dim), name="obs")
    s_next = tf.placeholder(dtype=tf.float32, shape=(None, *obs_dim), name="obs_next")
    if isinstance(act_space, gym.spaces.Box):
        a = tf.placeholder(dtype=tf.float32, shape=(None, *act_dim), name="action")
    else:
        a = tf.placeholder(dtype=tf.int32, shape=(None,), name="action")
    r = tf.placeholder(dtype=tf.float32, shape=None, name="rewards")
    termnt = tf.placeholder(dtype=tf.float32, shape=None)

    # Model
    with tf.variable_scope("main"):
        mu, pi, q1, q2, q1_pi, q2_pi, v, log_pi = actor_critic_fn(
            s, a, act_space, hid, hid, hid
        )
    # Attention: target_V is calculated for next state!
    with tf.variable_scope("targ"):
        _, _, _, _, _, _, v_targ, _ = actor_critic_fn(
            s_next, a, act_space, hid, hid, hid
        )

    # Buffer
    buffer = sac_buffer(buffer_size, obs_dim, act_dim, gamma)

    # targets
    target_q = tf.stop_gradient(
        r + gamma * (1 - termnt) * v_targ
    )  # size (batch_size, ) !!
    # Should use q1_pi, q2_pi as sampled fresh actions of current policy, instead of actions stored in buffer
    target_v = tf.stop_gradient(tf.minimum(q1_pi, q2_pi) - alpha * log_pi)
    # Losses
    q1_loss = tf.reduce_mean((q1 - target_q) ** 2)
    q2_loss = tf.reduce_mean((q2 - target_q) ** 2)
    v_loss = tf.reduce_sum((v - target_v) ** 2)
    pi_loss = -tf.reduce_mean(q1_pi - alpha * log_pi)

    pi_opt = tf.train.AdamOptimizer(pi_lr).minimize(
        pi_loss, var_list=tf.trainable_variables(scope="main/pi")
    )
    q1_opt = tf.train.AdamOptimizer(q_lr).minimize(
        q1_loss, var_list=tf.trainable_variables(scope="main/q1")
    )
    q2_opt = tf.train.AdamOptimizer(q_lr).minimize(
        q2_loss, var_list=tf.trainable_variables(scope="main/q2")
    )
    v_opt = tf.train.AdamOptimizer(v_lr).minimize(
        v_loss, var_list=tf.trainable_variables(scope="main/v")
    )

    # Target update
    var_main = tf.trainable_variables(scope="main")
    var_targ = tf.trainable_variables(scope="targ")
    target_update = tf.group(
        [
            tf.assign(v_targ, rho * v_targ + (1 - rho) * v)
            for v_targ, v in zip(var_targ, var_main)
        ]
    )

    # Target initialization
    target_int = tf.group(
        [tf.assign(v_targ, v_main) for v_targ, v_main in zip(var_targ, var_main)]
    )

    # Number of variables
    var_pi = tf.trainable_variables(scope="main/pi")
    var_q1 = tf.trainable_variables(scope="main/q1")
    var_q2 = tf.trainable_variables(scope="main/q2")
    var_v = tf.trainable_variables(scope="main/v")

    num_pi = 0
    num_q1 = 0
    num_q2 = 0
    num_v = 0

    for v in var_pi:
        num_pi += np.prod(v.shape)
    for v in var_q1:
        num_q1 += np.prod(v.shape)
    for v in var_q2:
        num_q2 += np.prod(v.shape)
    for v in var_v:
        num_v += np.prod(v.shape)

    tf.logging.info(
        "Number of trainable variables: pi {}, q1 {}, q2 {}, v {}".format(
            num_pi, num_q1, num_q2, num_v
        )
    )

    all_phs = [s, a, r, s_next, termnt]
    start_time = time.time()

    def test_agent(n=10):

        for _ in range(n):
            step = 0
            ob = env.reset()
            r = 0
            es_ret = 0
            done = False
            while (not done) and step < steps_per_episode:
                a = sess.run(mu, feed_dict={s: ob.reshape(1, -1)})

                ob, r, done, _ = env.step(a[0])
                es_ret += r
                step += 1
            logger.store(TestEpRet=es_ret, TestEpLen=step)

    # Training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(target_int)
        logger.setup_tf_saver(
            sess,
            inputs={"x": s, "a": a},
            outputs={"pi": pi, "q1": q1, "q2": q2, "v": v},
        )

        for ep in range(epoch):
            for es in range(episode):
                ob = env.reset()  # initial state
                r_t = 0
                es_ret = 0
                es_len = 0

                for _ in range(steps_per_episode):
                    if buffer.size < start_steps:
                        a_t = env.action_space.sample()
                        ob_next, r_t, done, _ = env.step(a_t)
                    else:
                        a_t = sess.run(pi, feed_dict={s: ob.reshape(1, -1)})
                        ob_next, r_t, done, _ = env.step(a_t[0])
                    es_ret += r_t
                    es_len += 1
                    # Ignore done if game is terminated by length. Done signal is only meaningful if meeting the terminal state
                    done = False if es_len == steps_per_episode else done

                    buffer.add(ob, a_t, r_t, ob_next, done)
                    ob = ob_next

                    # Updating
                    # Tip: Do not seperate training of policy and Q networks. Using the same samples will improve
                    # the learning process
                    if done or es_len == steps_per_episode:
                        for i in range(es_len):
                            batch_tuple = buffer.sample(batch_size)

                            inputs_minbatch = {
                                k: v for k, v in zip(all_phs, batch_tuple)
                            }
                            q1_ls, q2_ls, q1_val, q2_val, log_pi_val, _, _ = sess.run(
                                [q1_loss, q2_loss, q1, q2, log_pi, q1_opt, q2_opt],
                                feed_dict=inputs_minbatch,
                            )
                            v_ls, v_val, _ = sess.run(
                                [v_loss, v, v_opt], feed_dict=inputs_minbatch
                            )
                            pi_ls, _, = sess.run(
                                [pi_loss, pi_opt], feed_dict=inputs_minbatch
                            )

                            sess.run(target_update, feed_dict=inputs_minbatch)
                            # print(
                            #     sess.run(tf.shape(target_q), feed_dict=inputs_minbatch)
                            # )
                            # print(
                            #     sess.run(tf.shape(target_v), feed_dict=inputs_minbatch)
                            # )

                            logger.store(
                                LossQ1=q1_ls,
                                LossQ2=q2_ls,
                                LossV=v_ls,
                                LossPi=pi_ls,
                                Q1Val=q1_val,
                                Q2Val=q2_val,
                                VVal=v_val,
                                Log_pi=log_pi_val,
                            )

                        logger.store(EpLen=es_len, EpRet=es_ret)

                    if done:
                        ob = env.reset()
                        r_t = 0
                        es_ret = 0
                        es_len = 0

            # Save info
            # Save model every epoch
            logger.save_state({"env": env})

            # Test performance

            test_agent()

            # Logger
            logger.log_tabular("Epoch", ep)
            logger.log_tabular(
                "TotalEnvInteracts", (ep + 1) * episode * steps_per_episode
            )
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossQ1", average_only=True)
            logger.log_tabular("LossQ2", average_only=True)
            logger.log_tabular("LossV", average_only=True)

            logger.log_tabular("Q1Val", with_min_and_max=True)
            logger.log_tabular("Q2Val", with_min_and_max=True)
            logger.log_tabular("VVal", with_min_and_max=True)
            logger.log_tabular("Log_pi", with_min_and_max=True)

            logger.log_tabular("TestEpRet", with_min_and_max=True)
            logger.log_tabular("TestEpLen", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for sac")
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--pi_lr", type=float, default=0.001)
    parser.add_argument("--v_lr", type=float, default=0.001)
    parser.add_argument("--q_lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--hid", type=int, nargs="+", default=[300, 300])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--exp_name", type=str, default="sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.995)
    parser.add_argument("--alpha", type=float, default=0.1)

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # If directly return the gym env object, will cause stack overflow. Should return the function pointer

    sac(
        args.seed,
        lambda: gym.make(args.env),
        actor_critic,
        args.epoch,
        args.episode,
        args.steps_per_episode,
        args.pi_lr,
        args.q_lr,
        args.v_lr,
        args.gamma,
        args.hid,
        args.buffer_size,
        args.batch_size,
        logger_kwargs,
        args.rho,
        args.alpha,
    )

