import multiprocessing as mp
from multiprocessing import Queue, Process, Value
import tensorflow as tf
import numpy as np
from spinup.utils.logx import EpochLogger
import gym
import argparse

EPS = 1e-8

tf.logging.set_verbosity(tf.logging.INFO)


def log_p_gaussian(x, mu, logstd):
    return tf.reduce_sum(
        -0.5
        * ((x - mu) ** 2 / (tf.exp(logstd) + EPS) ** 2 + tf.log(2 * np.pi) + 2 * logstd)
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
        with tf.variable_scope("common"):
            com = mlp(
                s,
                self._hid[:-1],
                self._hid[-1],
                activation=tf.nn.tanh,
                output_activation=tf.tanh,
            )
        with tf.variable_scope("pi"):
            logits = mlp(
                com, None, self._action_dim, activation=None, output_activation=None
            )
            logp_all = tf.nn.log_softmax(logits)  # batch_size x action_dim, [n ,m]
            pi = tf.squeeze(
                tf.multinomial(logits, 1)
            )  # batch_size x action_index, [n, 1]
            logp_pi = tf.reduce_sum(
                tf.one_hot(pi, depth=self._action_dim) * logp_all, axis=1
            )  # log probability of policy action at current state
            logp = tf.reduce_sum(
                tf.one_hot(a, depth=self._action_dim) * logp_all, axis=1
            )  # log probability of action a at current state
        return com, pi, logp, logp_pi


class mlp_with_diagonal_gaussian:
    """
    Diagonal Gaussian policy suitable for discrete and continous actions
    """

    def __init__(self, policy_hid, action_dim):
        self._hid = policy_hid
        self._action_dim = action_dim
        self.logstd = tf.get_variable(
            "log_std", initializer=-0.5 * np.ones(action_dim, dtype=tf.float32)
        )

    def run(self, s, a):
        with tf.variable_scope("common"):
            com = mlp(
                s,
                self._hid[:-1],
                self._hid[-1],
                activation=tf.nn.tanh,
                output_activation=tf.tanh,
            )
        with tf.variable_scope("pi"):
            mu = mlp(com, None, self._action_dim, None, None)
            std = tf.exp(self.logstd)
            pi = tf.random_normal(tf.shape(mu)) * std + mu
            logp = log_p_gaussian(a, mu, self.logstd)
            logp_pi = log_p_gaussian(pi, mu, self.logstd)
        return com, pi, logp, logp_pi


# Actor-critic
def actor_critic(action_space, s, a, common_hid):

    with tf.variable_scope("model"):
        if isinstance(action_space, gym.spaces.Box):
            # Continuous action
            actor = mlp_with_diagonal_gaussian(common_hid, action_space.shape)
        else:
            # Discrete action
            actor = mlp_with_categorical(common_hid, action_space.n)
        com, pi, logp, logp_pi = actor.run(s, a)
        with tf.variable_scope("v"):
            v = mlp(com, None, 1, None, None)

    return pi, logp, logp_pi, v


# Operations in seperate worker(single process)
def a3c_worker():
    pass


def a3c(
    seed,
    env_name,
    epoch,
    episode,
    steps_per_episode,
    pi_lr,
    v_lr,
    gamma,
    hid,
    logger_kwargs,
    numProc,
    ps_host,
    worker_host,
    job_nm,
    task_ind,
):
    # Cluster
    cluster = tf.train.ClusterSpec({"ps": ps_host, "worker": worker_host})

    # Server
    server = tf.train.Server(cluster, job_name=job_nm, task_index=task_ind)

    # Entry-point of A3C
    # Global parameters
    if job_nm == "ps":
        server.join()
    # Local workers
    else:
        with tf.device(
            tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_ind, cluster=cluster
            )
        ):
            # Env
            env = gym.make(env_name)
            act_space = env.action_space
            act_dim = env.action_space.shape
            act_ub = act_space.high[0]
            act_lb = act_space.low[0]
            obs_dim = env.observation_space.shape

            # Placeholders
            s = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name="s")
            a = tf.placeholder(dtype=tf.float32, shape=(None, act_dim), name="a")
            r = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="r")
            ret = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="return")

            # Actor Critic model
            pi, logp, logp_pi, v = actor_critic(act_space, s, a, hid)

            # Local params
            var_com = tf.local_variables(scope="model/common")
            var_pi = tf.local_variables(scope="model/pi")
            var_v = tf.local_variables(scope="model/v")

            num_com = 0
            num_pi = 0
            num_v = 0

            for v in var_com:
                num_v += np.prod(v.shape)
            for v in var_pi:
                num_pi += np.prod(v.shape)
            for v in var_v:
                num_v += np.prod(v.shape)
            num_var = num_v + num_com + num_pi

            tf.logging.info(
                "overall var {} common {} pi {} v {}".format(
                    num_var, num_com, num_pi, num_v
                )
            )

            # Losses
            pi_loss = tf.reduce_mean(logp * (ret - v))
            v_loss = tf.reduce_mean((v - ret) ** 2)

            # Optimizers
            pi_grad = tf.train.RMSPropOptimizer(learning_rate=pi_lr).compute_gradients(
                loss=pi_loss
            )
            v_grad = tf.train.RMSPropOptimizer(learning_rate=v_lr).compute_gradients(
                loss=v_loss
            )

        # Training in worker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="arguments for a3c, distributed tensorflow version"
    )
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--pi_lr", type=float, default=0.001)
    parser.add_argument("--v_lr", type=float, default=0.001)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--episode", type=int, default=4)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--hid", type=int, nargs="+", default=[300, 300])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exp_name", type=str, default="a3c")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--numProc", type=int, default=3)

    parser.add_argument("--ps_host", type=str, default="127.0.0.1:12222")
    parser.add_argument(
        "--worker_host",
        type=str,
        nargs="+",
        default=["127.0.0.1:12223", "127.0.0.1:12224"],
    )
    parser.add_argument("--job_nm", type=str, default="ps")
    parser.add_argument("--task_ind", type=int, default=0)

    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # If directly return the gym env object, will cause stack overflow. Should return the function pointer

    a3c(
        args.seed,
        args.env,
        args.epoch,
        args.episode,
        args.steps_per_episode,
        args.pi_lr,
        args.v_lr,
        args.gamma,
        args.hid,
        logger_kwargs,
        args.numProc,
        args.ps_host,
        args.worker_host,
        args.job_nm,
        args.task_ind,
    )
