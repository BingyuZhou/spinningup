import tensorflow as tf
import numpy as np
import gym
import tqdm
import argparse
import scipy.signal
import time
from spinup.utils.logx import EpochLogger

EPS = 1e-8

tf.logging.set_verbosity(tf.logging.INFO)


def mlp(x, hid, output_layer, activation, output_activation):
    input = x
    for h in hid:
        input = tf.layers.dense(input, h, activation)
    return tf.layers.dense(input, output_layer, output_activation)


def actor_critic(s, a, action_space, policy_hid=[64, 64], q_hid=[64]):
    """
    Actor Critic model:
    Inputs: observation, reward
    Outputs: action, logp_pi, logp, v
    """
    with tf.variable_scope("pi"):
        pi = action_space.high[0] * mlp(
            s,
            policy_hid,
            action_space.shape,
            activation=tf.nn.relu,
            output_activation=tf.nn.tanh,
        )

    with tf.variable_scope("q"):
        q = mlp(
            tf.concat([s, a], 1),
            q_hid,
            1,
            activation=tf.nn.relu,
            output_activation=None,
        )

        # q_pi is used for the target, approximate max Q(s,a)
        q_pi = mlp(
            tf.concat([s, pi], 1),
            q_hid,
            1,
            activation=tf.nn.relu,
            output_activation=None,
        )

    return pi, q, q_pi


class ddpg_buffer:
    """
    Replay Buffer (On policy)
    - Only contain the transitions from the current policy
    - overwritten when a new policy is applied
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


def ddpg(
    seed,
    env_fn,
    actor_critic_fn,
    epoch,
    episode,
    steps_per_episode,
    pi_lr,
    q_lr,
    gamma,
    hid,
    buffer_size,
    batch_size,
    pi_train_itr,
    q_train_itr,
    logger_kwargs,
    rho,
):
    """
    Deep Deterministic Policy Gradient
    with tricks of target networks
    - Off poliocy
    - Only for continous action space
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
    a = tf.placeholder(dtype=tf.float32, shape=(None, *act_dim), name="action")
    r = tf.placeholder(dtype=tf.float32, shape=None, name="rewards")
    termnt = tf.placeholder(dtype=tf.float32, shape=None)

    # Check continuous action space
    assert isinstance(env.action_space, gym.spaces.Box)

    # Model
    with tf.variable_scope("main"):
        pi, q, q_pi = actor_critic_fn(s, a, act_space, hid, hid)
    with tf.variable_scope("targ"):
        pi_targ, _, q_targ = actor_critic_fn(s, a, act_space, hid, hid)

    # Buffer
    buffer = ddpg_buffer(buffer_size, obs_dim, act_dim, gamma)
    # Losses
    target = r + gamma * (1 - termnt) * q_targ
    q_loss = tf.reduce_mean((q - target) ** 2)
    pi_loss = -tf.reduce_mean(q_pi)

    var_main = tf.trainable_variables(scope="main")
    var_targ = tf.trainable_variables(scope="targ")
    pi_opt = tf.train.AdamOptimizer(pi_lr).minimize(
        pi_loss, var_list=tf.trainable_variables(scope="main/pi")
    )
    q_opt = tf.train.AdamOptimizer(q_lr).minimize(
        q_loss, var_list=tf.trainable_variables(scope="main/q")
    )
    # Target update
    target_update = tf.group(
        [
            tf.assign(v_targ, rho * v_targ + (1 - rho) * v)
            for v_targ, v in zip(var_targ, var_main)
        ]
    )

    # Number of variables
    var_pi = tf.trainable_variables(scope="main/pi")
    var_q = tf.trainable_variables(scope="main/q")
    num_pi = 0
    num_v = 0
    for v in var_pi:
        num_pi += np.prod(v.shape)
    for v in var_q:
        num_v += np.prod(v.shape)

    tf.logging.info("Number of trainable variables: pi {}, q {}".format(num_pi, num_v))

    all_phs = [s, a, r, s_next, termnt]
    start_time = time.time()
    # Training
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        logger.setup_tf_saver(sess, inputs={"x": s}, outputs={"pi": pi, "v": v})
        for ep in range(epoch):
            es_len = 0
            es_len_prev = 0
            for es in range(episode):
                ob = env.reset()  # initial state
                r_t = 0
                es_ret = 0

                for step in range(steps_per_episode):
                    a_t, v_t, logp_t = sess.run(
                        [pi, v, logp_pi], feed_dict={s: ob.reshape(1, -1)}
                    )
                    buffer.add(ob, a_t, r_t, v_t, logp_t, es_len)
                    ob, r_t, done, _ = env.step(a_t[0])

                    es_ret += r_t
                    es_len += 1

                    if done or step == steps_per_episode - 1:
                        if done:
                            buffer.final(v=r_t)
                            logger.store(EpRet=es_ret, EpLen=es_len - es_len_prev)

                        else:
                            buffer.final(sess.run(v, feed_dict={s: ob.reshape(1, -1)}))
                        ob = env.reset()
                        r_t = 0
                        es_ret = 0
                        es_len_prev = es_len
            buffer.normalize_adv()
            batch_tuple_all = buffer.sample(episode * steps_per_episode)

            inputs = {k: v for k, v in zip(all_phs, batch_tuple_all)}
            pi_loss_old, v_loss_old = sess.run([pi_loss, v_loss], feed_dict=inputs)

            # Update policy
            for _ in range(pi_train_itr):
                batch_tuple = buffer.sample(batch_size)
                inputs_minbatch = {k: v for k, v in zip(all_phs, batch_tuple)}
                _, kl = sess.run([pi_opt, approx_kl], feed_dict=inputs_minbatch)
                if np.mean(kl) > 1.5 * target_kl:
                    tf.logging.info("Early stop policy update since KL too large")
                    break

            for _ in range(v_train_itr):
                # Update value function
                batch_tuple = buffer.sample(batch_size)
                sess.run(v_opt, feed_dict={r_to_go: batch_tuple[2], s: batch_tuple[0]})

            pi_loss_new, v_loss_new, approx_entropy_v, kl = sess.run(
                [pi_loss, v_loss, approx_entropy, approx_kl], feed_dict=inputs
            )

            # Save model
            if (ep % 10 == 0) or (ep == epoch - 1):
                logger.save_state({"env": env})

            # Log
            logger.store(
                LossPi=pi_loss_old,
                LossV=v_loss_old,
                DeltaLossPi=pi_loss_new - pi_loss_old,
                DeltaLossV=v_loss_new - v_loss_old,
                Entropy=approx_entropy_v,
                KL=kl,
            )

            logger.log_tabular("Epoch", ep)
            logger.log_tabular(
                "TotalEnvInteracts", (ep + 1) * episode * steps_per_episode
            )
            logger.log_tabular("EpLen", average_only=True)
            logger.log_tabular("EpRet", with_min_and_max=True)
            logger.log_tabular("LossPi", average_only=True)
            logger.log_tabular("LossV", average_only=True)
            logger.log_tabular("DeltaLossPi", average_only=True)
            logger.log_tabular("DeltaLossV", average_only=True)
            logger.log_tabular("Entropy", average_only=True)
            logger.log_tabular("KL", average_only=True)
            logger.log_tabular("Time", time.time() - start_time)
            logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="arguments for vpg")
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--pi_lr", type=float, default=0.0003)
    parser.add_argument("--q_lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--hid", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=4001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--pi_train_itr", type=int, default=120)
    parser.add_argument("--q_train_itr", type=int, default=80)
    parser.add_argument("--exp_name", type=str, default="ddpg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rho", type=float, default=0.9)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # If directly return the gym env object, will cause stack overflow. Should return the function pointer

    ddpg(
        args.seed,
        lambda: gym.make(args.env),
        actor_critic,
        args.epoch,
        args.episode,
        args.steps_per_episode,
        args.pi_lr,
        args.q_lr,
        args.gamma,
        args.hid,
        args.buffer_size,
        args.batch_size,
        args.pi_train_itr,
        args.q_train_itr,
        logger_kwargs,
        args.rho,
    )

