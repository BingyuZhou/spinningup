import tensorflow as tf
import numpy as np
import gym
import argparse
import scipy.signal
import os
import json

EPS = 1e-8

tf.logging.set_verbosity(tf.logging.WARN)


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
        self._rescale = 255.0

    def run(self, s, a):
        with tf.variable_scope("common"):
            layer1 = tf.layers.conv2d(
                s / self._rescale, filters=32, kernel_size=8, strides=4
            )
            layer1_re = tf.nn.relu(layer1)
            layer2 = tf.layers.conv2d(layer1_re, filters=64, kernel_size=4, strides=2)
            layer2_re = tf.nn.relu(layer2)
            layer3 = tf.layers.conv2d(layer2_re, filters=64, kernel_size=3, strides=1)
            layer3_re = tf.nn.relu(layer3)
            layer_flat = tf.layers.flatten(layer3_re)
            com = tf.layers.dense(layer_flat, 512, activation=tf.nn.relu)

        with tf.variable_scope("pi"):
            logits = tf.layers.dense(com, self._action_dim)
            logp_all = tf.nn.log_softmax(logits)  # batch_size x action_dim, [n ,m]
            pi = tf.squeeze(
                tf.multinomial(logits, 1), axis=1
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

    def __init__(self, policy_hid, action_dim, LOG_STD_MAX, LOG_STD_MIN):
        self._hid = policy_hid
        self._action_dim = action_dim
        self._LOG_STD_MAX = LOG_STD_MAX
        self._LOG_STD_MIN = LOG_STD_MIN

    def run(self, s, a):
        with tf.variable_scope("common"):
            com = mlp(
                s,
                self._hid[:-1],
                self._hid[-1],
                activation=tf.nn.tanh,
                output_activation=tf.tanh,
            )

            log_std = tf.layers.dense(com, *self._action_dim, activation=tf.tanh)
            # Mapping from (-1,1) to (LOG_STD_MIN, LOG_STD_MAX)
            log_std = 0.5 * (self._LOG_STD_MAX - self._LOG_STD_MIN) * log_std + 0.5 * (
                self._LOG_STD_MAX + self._LOG_STD_MIN
            )
        with tf.variable_scope("pi"):
            mu = mlp(com, [], *self._action_dim, None, None)
            std = tf.exp(log_std)
            pi = tf.random_normal(tf.shape(mu)) * std + mu
            logp = log_p_gaussian(a, mu, log_std)
            logp_pi = log_p_gaussian(pi, mu, log_std)
        return com, pi, logp, logp_pi


# Actor-critic
def actor_critic(action_space, s, a, common_hid, LOG_STD_MAX, LOG_STD_MIN):

    with tf.variable_scope("model"):
        if isinstance(action_space, gym.spaces.Box):
            # Continuous action
            actor = mlp_with_diagonal_gaussian(
                common_hid, action_space.shape, LOG_STD_MAX, LOG_STD_MIN
            )
        else:
            # Discrete action
            actor = mlp_with_categorical(common_hid, action_space.n)
        com, pi, logp, logp_pi = actor.run(s, a)
        with tf.variable_scope("v"):
            v = tf.layers.dense(com, 1)
    return pi, logp, logp_pi, v


def cum_discounted_sum(x, discount):
    """
        Magic formula to compute cummunated sum of discounted value (Faster)
        Input: x = [x1, x2, x3]
        Output: x1+discount*x2+discount^2*x3, x2+discount*x3, x3
        """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# Operations in seperate worker(single process)
def a3c_worker(sess, env, s, pi, v, steps_per_episode, gamma):
    # Sampling in local environment, collecting rewards-to-go and logp
    ob = env.reset()
    done = False
    r_t_buffer = []
    state_buffer = []
    v_buffer = []
    action_buffer = []
    ep_len = 0

    while not done and ep_len < steps_per_episode:
        a_t, v_t = sess.run([pi, v], feed_dict={s: np.expand_dims(ob, 0)})
        state_buffer.append(ob)
        action_buffer.append(a_t[0])
        v_buffer.append(v_t[0][0])

        ob, r_t, done, _ = env.step(a_t[0])
        r_t_buffer.append(r_t)
        ep_len += 1

    # Booststrap final state
    if done:
        r_t_buffer.append(0)
    else:
        r_t_buffer.append(sess.run(v, feed_dict={s: np.expand_dims(ob, 0)}))

    # rewards-to-go
    r_to_go = cum_discounted_sum(np.asanyarray(r_t_buffer), gamma)[:-1]

    # adv
    adv_buffer = np.array(r_to_go) - np.array(v_buffer)

    return (
        ep_len,
        sum(r_t_buffer[:-1]),
        np.asanyarray(r_to_go),
        np.asanyarray(state_buffer),
        np.asanyarray(action_buffer),
        np.asanyarray(adv_buffer),
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
    cluster,
    job_name,
    task_index,
    alpha=0.01,
):
    # Cluster
    cluster_spec = tf.train.ClusterSpec(cluster)

    # Server
    server = tf.train.Server(cluster_spec, job_name=job_name, task_index=task_index)

    # Entry-point of A3C
    # Global parameters
    if job_name == "ps":
        server.join()
    else:
        # Global graph
        with tf.device("/job:ps/task:0"):
            # Global Env
            env_g = gym.make(env_name)

            act_space = env_g.action_space
            act_dim = env_g.action_space.shape
            obs_dim = env_g.observation_space.shape

            # Shared global params
            global_step = tf.Variable(
                name="global_step", initial_value=0, dtype=tf.int32
            )
            LOG_STD_MAX = 2
            LOG_STD_MIN = -20

            # Placeholders
            s_g = tf.placeholder(dtype=tf.float32, shape=(None, *obs_dim), name="s")
            if isinstance(act_space, gym.spaces.Box):
                a_g = tf.placeholder(dtype=tf.float32, shape=(None, *act_dim), name="a")
            else:
                a_g = tf.placeholder(dtype=tf.int32, shape=(None,), name="a")

            # Actor Critic model
            with tf.variable_scope("global"):
                pi_g, logp_g, logp_pi_g, v_g = actor_critic(
                    act_space, s_g, a_g, hid, LOG_STD_MAX, LOG_STD_MIN
                )

            # Global params
            var_com_g = tf.trainable_variables(scope="global/model/common")
            var_pi_g = tf.trainable_variables(scope="global/model/pi")
            var_v_g = tf.trainable_variables(scope="global/model/v")

            var_com_pi_g = var_com_g + var_pi_g
            var_com_v_g = var_com_g + var_v_g

        # Local graph
        # Tips: when using tf.train.replica_device_setter(), all the variables (mainly weights of networks)
        # are placed in ps taskes by default. Other operations and states are placed in work_device, which means there is NO local copy of variables!!
        with tf.device("/job:worker/task:%d" % task_index):
            # Env
            env = gym.make(env_name)

            # Placeholders
            s = tf.placeholder(dtype=tf.float32, shape=(None, *obs_dim), name="s")
            if isinstance(act_space, gym.spaces.Box):
                a = tf.placeholder(dtype=tf.float32, shape=(None, *act_dim), name="a")
            else:
                a = tf.placeholder(dtype=tf.int32, shape=(None,), name="a")

            ret = tf.placeholder(dtype=tf.float32, shape=(None,), name="return")
            adv = tf.placeholder(dtype=tf.float32, shape=(None,), name="advantage")

            # Actor Critic model
            pi, logp, logp_pi, v = actor_critic(
                act_space, s, a, hid, LOG_STD_MAX, LOG_STD_MIN
            )

            # Local params
            var_com = tf.trainable_variables(scope="model/common")
            var_pi = tf.trainable_variables(scope="model/pi")
            var_v = tf.trainable_variables(scope="model/v")

            num_com = 0
            num_pi = 0
            num_v = 0

            for var in var_com:
                num_com += np.prod(var.shape)
            for var in var_pi:
                num_pi += np.prod(var.shape)
            for var in var_v:
                num_v += np.prod(var.shape)
            num_var = num_v + num_com + num_pi

            tf.logging.info(
                "overall var {} common {} pi {} v {}".format(
                    num_var, num_com, num_pi, num_v
                )
            )

            # Losses with entropy
            pi_loss = -tf.reduce_mean(logp * adv - alpha * logp_pi)
            v_loss = tf.reduce_mean((v - ret) ** 2)

            # Optimizers
            pi_opt = tf.train.RMSPropOptimizer(learning_rate=pi_lr, decay=0.99)
            pi_grad = tf.gradients(pi_loss, var_com + var_pi)
            v_opt = tf.train.RMSPropOptimizer(learning_rate=v_lr, decay=0.99)
            v_grad = tf.gradients(v_loss, var_com + var_v)

        # Asyn update global params
        asyn_pi_update = pi_opt.apply_gradients(
            [(grad, var) for grad, var in zip(pi_grad, var_com_pi_g)]
        )
        asyn_v_update = v_opt.apply_gradients(
            [(grad, var) for grad, var in zip(v_grad, var_com_v_g)]
        )

        # Sync gloabl params -> local params
        sync_local_params = tf.group(
            [
                tf.assign(v, v_g)
                for v, v_g in zip(
                    var_com + var_pi + var_v, var_com_g + var_pi_g + var_v_g
                )
            ]
        )

        def test(sess, n=10):
            ep_ret = 0
            for _ in range(n):
                step = 0
                ob = env_g.reset()
                r_t = 0
                done = False
                while (not done) and step < steps_per_episode:
                    a_t = sess.run(pi, feed_dict={s: np.expand_dims(ob, 0)})
                    ob, r_t, done, _ = env_g.step(a_t[0])
                    ep_ret += r_t
                    step += 1
            return ep_ret / n

        episode_ret = tf.placeholder(dtype=tf.float32, shape=[], name="episode_ret")
        tf.summary.scalar("ep_ret", episode_ret)
        test_ep_ret = tf.placeholder(
            dtype=tf.float32, shape=[], name="test_episode_ret"
        )
        tf.summary.scalar("test_ep_ret", test_ep_ret)

        merge_tb = tf.summary.merge_all()

        # Training in worker
        with tf.Session(server.target) as sess:
            writer = tf.summary.FileWriter("./summary/", sess.graph)

            sess.run(tf.global_variables_initializer())

            global_step_t = sess.run(global_step)
            while global_step_t < max_step:
                sess.run(sync_local_params)

                ep_len, ep_ret, r_to_go, state_buffer, action_buffer, adv_buffer = a3c_worker(
                    sess, env, s, pi, v, steps_per_episode, gamma
                )
                sess.run(tf.assign(global_step, global_step + ep_len))
                global_step_t = sess.run(global_step)

                _, _, ls_v = sess.run(
                    [v_grad, asyn_v_update, v_loss],
                    feed_dict={s: state_buffer, a: action_buffer, ret: r_to_go},
                )
                _, _, ls_pi = sess.run(
                    [pi_grad, asyn_pi_update, pi_loss],
                    feed_dict={s: state_buffer, a: action_buffer, adv: adv_buffer},
                )
                # log in chief node
                if job_name == "worker" and task_index == 0:
                    test_ret = None
                    if global_step_t % 200 <= 50:
                        test_ret = test(sess)

                    summary = sess.run(
                        merge_tb, feed_dict={episode_ret: ep_ret, test_ep_ret: test_ret}
                    )
                    writer.add_summary(summary, global_step=global_step_t)
            tf.logging.warn(
                "process {} is done at step {}".format(task_index, global_step_t)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="arguments for a3c, distributed tensorflow version"
    )
    parser.add_argument("--env", type=str, default="Breakout-v0")
    parser.add_argument("--pi_lr", type=float, default=0.001)
    parser.add_argument("--v_lr", type=float, default=0.001)
    parser.add_argument("--max_step", type=int, default=5e5)
    parser.add_argument("--steps_per_episode", type=int, default=20)
    parser.add_argument("--hid", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--exp_name", type=str, default="a3c")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # If directly return the gym env object, will cause stack overflow. Should return the function pointer

    """Parse TF_CONFIG to cluster_spec and call run() method.

    TF_CONFIG environment variable is available when running using
    gcloud either locally or on cloud. It has all the information required
    to create a ClusterSpec which is important for running distributed code.

    Args:
        args (args): Input arguments.
    """

    tf_config = os.environ.get("TF_CONFIG")

    tf_config_json = json.loads(tf_config)
    cluster = tf_config_json.get("cluster")
    job_name = tf_config_json.get("task", {}).get("type")
    task_index = tf_config_json.get("task", {}).get("index")

    a3c(
        args.seed,
        args.env,
        args.max_step,
        args.steps_per_episode,
        args.pi_lr,
        args.v_lr,
        args.gamma,
        args.hid,
        cluster,
        job_name,
        task_index,
    )
