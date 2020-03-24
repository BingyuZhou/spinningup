import tensorflow as tf
from tensorflow import keras
import tensorboard as tb
import numpy as np
import gym
import datetime

tf.keras.backend.set_floatx('float32')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
test_log_dir = 'logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


class Policy(keras.Model):
    def __init__(self, act_size):
        super(Policy, self).__init__()

        self.layer1 = keras.layers.Dense(256, activation=tf.tanh)
        self.layer2 = keras.layers.Dense(100, activation=tf.tanh)
        self.layer3 = keras.layers.Dense(act_size)

        self.act_size = act_size

    # action distribution
    def call(self, obs):
        x = self.layer1(obs)
        x = self.layer2(x)
        x = self.layer3(x)
        logp = tf.nn.log_softmax(x)
        return logp

    # action sampling
    def sample(self, obs):
        return tf.squeeze(tf.random.categorical(self.call(obs), 1), axis=1)

    def loss(self, obs, act, ret):
        # expection grad log
        mask = tf.one_hot(act, depth=self.act_size)
        logp_a = tf.reduce_sum(self.call(obs) * mask, axis=1)  # logp(a|s)
        loss = -tf.reduce_mean(logp_a * ret)
        return loss


def test(env_name, policy):
    ep_rew = []
    ep_len = 0

    env = gym.make(env_name)

    obs = env.reset()

    test_ret = []
    test_len = []

    done = False
    for _ in range(10):
        while not done:
            act = policy.sample(obs.reshape(1, -1))[0].numpy()

            obs, reward, done, _ = env.step(act)

            ep_rew.append(reward)
            ep_len += 1

        # compute return
        test_ret.append(np.sum(ep_rew))
        test_len.append(ep_len)

        # respawn env
        obs = env.reset()
        ep_len = 0
        ep_rew.clear()
        done = False

    return test_ret, test_len


# run one policy update
def train(env_name, batch_size, epochs):
    # set env
    env = gym.make(env_name)

    policy = Policy(env.action_space.n)

    optimizer = keras.optimizers.Adam()

    def train_one_epoch():
        # initialize replay buffer
        batch_obs = []
        batch_act = []  # batch action
        batch_ret = []  # batch return
        batch_len = []  # batch trajectory length
        ep_rew = []  # episode rewards (trajectory rewards)
        ep_len = 0  # length of trajectory

        # initial observation
        obs = env.reset()

        # render first episode of each epoch
        render_env = True

        # fill in recorded trajectories
        while True:
            if render_env:
                env.render()

            act = policy.sample(obs.reshape(1, -1))[0].numpy()

            batch_act.append(act)
            batch_obs.append(obs.copy())

            obs, reward, done, _ = env.step(act)

            ep_rew.append(reward)
            ep_len += 1

            if done:
                # compute return
                ret = np.sum(ep_rew)
                batch_ret += [ret] * ep_len
                batch_len.append(ep_len)

                # respawn env
                obs = env.reset()
                ep_len = 0
                ep_rew.clear()

                # stop render
                render_env = False

                if len(batch_obs) > batch_size:
                    break

        @tf.function
        def train_step(obs, act, ret):
            with tf.GradientTape() as tape:
                ls = policy.loss(obs, act, ret)
            grad = tape.gradient(ls, policy.trainable_variables)
            optimizer.apply_gradients(zip(grad, policy.trainable_variables))
            return ls

        # update policy
        batch_loss = train_step(tf.constant(batch_obs), np.array(batch_act),
                                tf.constant(batch_ret, dtype=tf.float32))

        return batch_loss, batch_ret, batch_len

    for i in range(epochs):
        batch_loss, batch_ret, batch_len = train_one_epoch()
        with train_summary_writer.as_default():
            tf.summary.scalar('batch_ret', np.mean(batch_ret), step=i)
            tf.summary.scalar('batch_len', np.mean(batch_len), step=i)

        print("epoch {0:2d} loss {1:.3f} batch_ret {2:.3f} batch_len {3:.3f}".
              format(i, batch_loss.numpy(), np.mean(batch_ret),
                     np.mean(batch_len)))

        # Test
        if i % 10 == 0:
            test_ret, test_len = test(env_name, policy)
            with test_summary_writer.as_default():
                tf.summary.scalar('test_ret', np.mean(test_ret), step=i)
                tf.summary.scalar('test_len', np.mean(test_len), step=i)


if __name__ == '__main__':
    env_name = 'CartPole-v1'
    epochs = 500
    batch_size = 1000
    train(env_name, batch_size, epochs)
