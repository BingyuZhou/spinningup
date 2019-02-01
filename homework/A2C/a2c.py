import multiprocessing as mp
from multiprocessing import Queue, Process, Value
import tensorflow as tf
import numpy as np
from spinup.utils.logx import EpochLogger
import gym
import argparse
import scipy

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


def cum_discounted_sum(x, discount):
    """
        Magic formula to compute cummunated sum of discounted value (Faster)
        Input: x = [x1, x2, x3]
        Output: x1+discount*x2+discount^2*x3, x2+discount*x3, x3
        """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# Operations in seperate worker(single process)
def a3c_worker(sess, ep_len, env, s, pi, v, global_step, steps_per_episode, gamma):
    # Sampling in local environment, collecting rewards-to-go and logp
    ob = env.reset()
    done = False
    r_t_buffer = []
    state_buffer = []
    v_buffer = []
    action_buffer = []

    while not done and ep_len < steps_per_episode:
        a_t, v_t = sess.run([pi, v], feed_dict={s: ob.reshape(1, -1)})
        state_buffer.append(ob.reshape(1, -1))
        action_buffer.append(a_t)
        v_buffer.append(v_t)

        ob, r_t, done, _ = env.step(a_t)

        r_t_buffer.append(r_t)
        ep_len += 1
        global_step += 1

    # Booststrap final state
    if done:
        r_t_buffer.append(0)
    else:
        r_t_buffer.append(sess.run(v, feed_dict={s: ob.reshape(1, -1)}))

    # rewards-to-go
    r_to_go = cum_discounted_sum(np.asanyarray(r_t_buffer), gamma)[:-1]

    return (
        global_step,
        r_to_go,
        np.asanyarray(state_buffer),
        np.asanyarray(action_buffer),
        np.asanyarray(v_buffer),
    )


def a3c(
    seed,
    env_name,
    max_step,
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
    else:
        # Local graph
        # Tips: when using tf.train.replica_device_setter(), all the variables (mainly weights of networks)
        # are placed in ps taskes by default. Other operations and states are placed in work_device, which means there is NO local copy of variables!!
        with tf.device("/job:worker/task:%d" % task_ind):
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
            pi_opt = tf.train.RMSPropOptimizer(learning_rate=pi_lr)
            pi_grad = pi_opt.compute_gradients(loss=pi_loss, var_list=var_com + var_pi)
            v_opt = tf.train.RMSPropOptimizer(learning_rate=v_lr)
            v_grad = v_opt.compute_gradients(loss=v_loss, var_list=var_com + var_v)

            # Local ep_len
            ep_len = 0

        # Global graph
        with tf.device("/job:ps/task:0"):
            # Global Env
            env_g = gym.make(env_name)

            # Placeholders
            s_g = tf.placeholder(dtype=tf.float32, shape=(None, obs_dim), name="s")
            a_g = tf.placeholder(dtype=tf.float32, shape=(None, act_dim), name="a")
            r_g = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="r")
            ret_g = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="return")

            # Actor Critic model
            pi_g, logp_g, logp_pi_g, v_g = actor_critic(act_space, s_g, a_g, hid)

            # Global params
            var_com_g = tf.trainable_variables(scope="model/common")
            var_pi_g = tf.trainable_variables(scope="model/pi")
            var_v_g = tf.trainable_variables(scope="model/v")

            var_com_pi_g = var_com_g + var_pi_g
            var_com_v_g = var_com_g + var_v_g

            # Global step
            global_step = 0

        # Asyn update global params
        with tf.device("/job:worker/task:%d" % task_ind):

            asyn_pi_update = pi_opt.apply_gradients(
                [(var_com_pi_g[i], pi_grad[i][1])] for i in len(pi_grad)
            )
            asyn_v_update = v_opt.apply_gradients(
                [(var_com_v_g[i], v_grad[i][1])] for i in len(pi_grad)
            )

            asyn_update = tf.group([asyn_pi_update, asyn_v_update])

        # Sync gloabl params -> local params
        with tf.device("/job:ps/task:0"):
            sync_local_params = tf.group(
                [
                    tf.assign(v, v_g)
                    for v, v_g in zip(
                        [var_com, var_pi, var_v], [var_com_g, var_pi_g, var_v_g]
                    )
                ]
            )

        # Training in worker

        with tf.Session(server.target) as sess:
            sess.run(tf.global_variables_initializer())
            while global_step < max_step:
                sess.run(sync_local_params)

                global_step, r_to_go, state_buffer, action_buffer, v_buffer = a3c_worker(
                    sess, ep_len, env, s, pi, v, global_step, steps_per_episode, gamma
                )

                v_grad_val, = sess.run(
                    v_grad, feed_dict={s: state_buffer, ret: r_to_go}
                )
                pi_grad_val = sess.run(
                    pi_grad,
                    feed_dict={
                        s: state_buffer,
                        a: action_buffer,
                        v: v_buffer,
                        ret: r_to_go,
                    },
                )

                sess.run(
                    asyn_update, feed_dict={pi_grad: pi_grad_val, v_grad: v_grad_val}
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="arguments for a3c, distributed tensorflow version"
    )
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--pi_lr", type=float, default=0.001)
    parser.add_argument("--v_lr", type=float, default=0.001)
    parser.add_argument("--max_step", type=int, default=1e6)
    parser.add_argument("--steps_per_episode", type=int, default=1e3)
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
        args.max_step,
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
