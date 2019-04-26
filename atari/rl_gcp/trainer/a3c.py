import tensorflow as tf
import numpy as np
import gym
import argparse
import scipy.signal
import os
import json

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

    def __init__(self, action_dim):
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

    def __init__(self, action_dim, LOG_STD_MAX, LOG_STD_MIN):
        self._action_dim = action_dim
        self._LOG_STD_MAX = LOG_STD_MAX
        self._LOG_STD_MIN = LOG_STD_MIN
        self._hid = []

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
def actor_critic(action_space, s, a, LOG_STD_MAX, LOG_STD_MIN):

    with tf.variable_scope("model"):
        if isinstance(action_space, gym.spaces.Box):
            # Continuous action
            actor = mlp_with_diagonal_gaussian(
                action_space.shape, LOG_STD_MAX, LOG_STD_MIN
            )
        else:
            # Discrete action
            actor = mlp_with_categorical(action_space.n)
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
    env_name,
    max_step,
    steps_per_episode,
    pi_lr,
    v_lr,
    gamma,
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
        tf.logging.info("ps stops...")
    elif job_name == "worker":
        tf.logging.info("entering worker ...")
        # Env
        env = gym.make(env_name)

        act_space = env.action_space
        act_dim = env.action_space.shape
        obs_dim = env.observation_space.shape

        LOG_STD_MAX = 2
        LOG_STD_MIN = -20

        # Local graph on worker node
        # Tips: when using tf.train.replica_device_setter(), all the variables (mainly weights of networks)
        # are placed in ps taskes by default. Other operations and states are placed in work_device, which means there is NO local copy of variables!!
        tf.logging.info(
            "building model {} task_ind {} ...".format(job_name, task_index)
        )
        with tf.device(
            tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % task_index, cluster=cluster_spec
            )
        ):
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
                act_space, s, a, LOG_STD_MAX, LOG_STD_MIN
            )

            # Local params
            var_com = tf.trainable_variables(scope="model/common")
            var_pi = tf.trainable_variables(scope="model/pi")
            var_v = tf.trainable_variables(scope="model/v")

            # Losses with entropy
            pi_loss = -tf.reduce_mean(logp * adv - alpha * logp_pi)
            tf.summary.scalar("pi_loss", pi_loss)
            v_loss = tf.reduce_mean((v - ret) ** 2)
            tf.summary.scalar("v_loss", v_loss)

            global_step = tf.train.get_or_create_global_step()

            # Optimizers
            pi_opt = tf.train.RMSPropOptimizer(learning_rate=pi_lr).minimize(
                loss=pi_loss, global_step=global_step, var_list=var_com + var_pi
            )

            v_opt = tf.train.RMSPropOptimizer(learning_rate=v_lr).minimize(
                loss=v_loss, global_step=global_step, var_list=var_com + var_v
            )

            episode_ret = tf.placeholder(dtype=tf.float32, shape=[], name="episode_ret")
            tf.summary.scalar("ep_ret", episode_ret)
            test_ep_ret = tf.placeholder(
                dtype=tf.float32, shape=[], name="test_episode_ret"
            )
            tf.summary.scalar("test_ep_ret", test_ep_ret)

            merge_tb = tf.summary.merge_all()

            tf.logging.info("finishing model...")

        # def test(sess, n=10):
        #     ep_ret = 0
        #     for _ in range(n):
        #         step = 0
        #         ob = env.reset()
        #         r_t = 0
        #         done = False
        #         while (not done) and step < steps_per_episode:
        #             a_t = sess.run(pi, feed_dict={s: np.expand_dims(ob, 0)})
        #             ob, r_t, done, _ = env.step(a_t[0])
        #             ep_ret += r_t
        #             step += 1
        #     return ep_ret / n

        hooks = [tf.train.StopAtStepHook(last_step=max_step)]

        # Training in worker
        tf.logging.info("start training...")
        writer = tf.summary.FileWriter("./summary/")
        tf.logging.info("summary ...")

        # with tf.train.MonitoredTrainingSession(
        #     master=server.target,
        #     is_chief=(task_index == 0),
        #     checkpoint_dir="train_logs",
        #     save_summaries_secs=None,
        #     save_summaries_steps=None,
        #     hooks=hooks,
        # ) as mon_sess:
        with tf.Session(server.target) as mon_sess:
            tf.logging.info(
                "global step {}".format(mon_sess.run(tf.train.get_global_step()))
            )
            mon_sess.run(tf.global_variables_initializer())
            while mon_sess.run(tf.train.get_global_step()) < max_step:
                ep_len, ep_ret, r_to_go, state_buffer, action_buffer, adv_buffer = a3c_worker(
                    mon_sess, env, s, pi, v, steps_per_episode, gamma
                )

                _, ls_v = mon_sess.run(
                    [v_opt, v_loss],
                    feed_dict={s: state_buffer, a: action_buffer, ret: r_to_go},
                )
                _, ls_pi = mon_sess.run(
                    [pi_opt, pi_loss],
                    feed_dict={s: state_buffer, a: action_buffer, adv: adv_buffer},
                )
                # log in chief node
                if task_index == 0:
                    test_ret = None
                    # if mon_sess.run(tf.train.get_global_step()) % 100 == 0:
                    #     test_ret = test(mon_sess)

                    summary = mon_sess.run(
                        merge_tb,
                        feed_dict={
                            episode_ret: ep_ret,
                            test_ep_ret: test_ret,
                            pi_loss: ls_pi,
                            v_loss: ls_v,
                        },
                    )
                    writer.add_summary(
                        summary, global_step=mon_sess.run(tf.train.get_global_step())
                    )

        tf.logging.info("training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="arguments for a3c, distributed tensorflow version"
    )
    parser.add_argument("--env", type=str, default="Breakout-v0")
    parser.add_argument("--pi_lr", type=float, default=0.001)
    parser.add_argument("--v_lr", type=float, default=0.001)
    parser.add_argument("--max_step", type=int, default=5e5)
    parser.add_argument("--steps_per_episode", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.99)

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
        args.env,
        args.max_step,
        args.steps_per_episode,
        args.pi_lr,
        args.v_lr,
        args.gamma,
        cluster,
        job_name,
        task_index,
    )
