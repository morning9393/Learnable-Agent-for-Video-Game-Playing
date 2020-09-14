import tkinter as tk
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.activations as ka
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.callbacks as kc

log_path = './log.txt'
backup_path = './backup/'


class Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = kl.Conv2D(filters=32,
                               kernel_size=(8, 8),
                               strides=4,
                               padding='valid',  # 'same'?
                               activation=ka.relu
                               )

        self.conv2 = kl.Conv2D(filters=64,
                               kernel_size=(4, 4),
                               strides=2,
                               padding='valid',  # 'same'?
                               activation=ka.relu
                               )

        self.conv3 = kl.Conv2D(filters=64,
                               kernel_size=(3, 3),
                               strides=1,
                               padding='valid',  # 'same'?
                               activation=ka.relu
                               )

        self.flat = kl.Flatten()
        self.h = kl.Dense(512, activation=ka.relu)
        self.actor = kl.Dense(num_actions, activation=None)
        self.critic = kl.Dense(1, activation=None)

    def call(self, inputs):
        x = tf.cast(inputs, dtype=tf.float32)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        h = self.h(x)
        a = self.actor(h)
        c = self.critic(h)
        return a, a, c


cur_epi = 0


class LearningRateReducerCb(kc.Callback):
    def __init__(self):
        super().__init__()
        self.lr = 0
        self.decay_step = 2000
        self.decay_rate = 0.75
        self.last_decay = 0

    def on_epoch_end(self, epoch, logs=None):
        self.lr = self.model.optimizer.lr
        if cur_epi % self.decay_step == 0 and cur_epi > self.last_decay:
            self.lr = self.lr * self.decay_rate
            self.model.optimizer.lr = self.lr
            self.last_decay = cur_epi


class A2C:
    def __init__(self, model, env, lr, num_episodes, gamma, n, coe, pre_weight_path=None, log=None, last_epi=0, queue=None):
        self.lr = lr
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.n = n
        self.coe = coe
        self.log = log
        self.last_epi = last_epi
        self.running = False
        self.queue = queue

        self.num_actions = 3
        self.env = env
        self.lifes = env.unwrapped.ale.lives()
        if self.lifes == 0:
            self.lifes = 1

        self.cb = LearningRateReducerCb()
        self.cp = kc.ModelCheckpoint(filepath=backup_path + 'model', save_weights_only=True)

        self.model = model
        optimizer = ko.RMSprop(lr=self.lr, rho=0.99, epsilon=1e-5)
        # losses = [self.actor_loss, self.critic_loss]
        losses = [self.actor_loss, self.actor_entropy, self.critic_loss]
        self.model.compile(optimizer=optimizer, loss=losses)
        if pre_weight_path is not None:
            self.model.load_weights(pre_weight_path + 'model')

        self.actor_losses = []
        self.entropies = []
        self.critic_losses = []

    def actor_loss(self, acts_advs, policy_pred):
        actions, advantages = tf.split(acts_advs, 2, axis=-1)
        weighted_sparse_ce = kls.SparseCategoricalCrossentropy(from_logits=True)
        actions = tf.cast(actions, tf.int32)
        policy_loss = weighted_sparse_ce(actions, policy_pred, sample_weight=advantages)
        return policy_loss

    def actor_entropy(self, acts_advs, policy_pred):
        probs = tf.nn.softmax(policy_pred)
        entropy = kls.categorical_crossentropy(probs, probs)
        return -1 * self.coe * entropy

    def critic_loss(self, v_cal, v_pred):
        critic_loss = 0.5 * kls.mean_squared_error(v_cal, v_pred)
        return critic_loss

    def train(self):
        self.running = True
        max_reward = -10000
        statis = 100
        statis_reward_sum = 0
        for episode in range(self.last_epi + 1, self.num_episodes):
            if not self.running:
                return
            global cur_epi
            cur_epi = episode
            real_reward = 0
            for i in range(self.lifes):
                if not self.running:
                    return
                reward = self.train_a_live()
                real_reward += reward
            if real_reward > max_reward:
                max_reward = real_reward

            statis_reward_sum += real_reward
            if episode % statis == 0:
                self.last_epi = episode
                average_actor_loss = np.mean(self.actor_losses)
                average_entropy = np.mean(self.entropies)
                average_critic_loss = np.mean(self.critic_losses)
                if self.log is not None:
                    self.log.insert(tk.END, "\n\nepisode: %d \n" % episode)
                    self.log.insert(tk.END, "learning rate: %s \n" % self.cb.lr)
                    self.log.insert(tk.END, "average reward: %s \n" % (statis_reward_sum / statis))
                    self.log.insert(tk.END, "max_reward: %d \n" % max_reward)
                    self.log.insert(tk.END, "average actor loss: %f \n" % average_actor_loss)
                    self.log.insert(tk.END, "average entropy: %f \n" % average_entropy)
                    self.log.insert(tk.END, "average critic loss: %f \n" % average_critic_loss)
                with open(log_path, 'a') as log:
                    log.write("\n\nepisode: %d \n" % episode)
                    log.write("learning rate: %s \n" % self.cb.lr)
                    log.write("average reward: %s \n" % (statis_reward_sum / statis))
                    log.write("max_reward: %d \n" % max_reward)
                    log.write("average actor loss: %f \n" % average_actor_loss)
                    log.write("average entropy: %f \n" % average_entropy)
                    log.write("average critic loss: %f \n" % average_critic_loss)
                statis_reward_sum = 0
                self.actor_losses = []
                self.entropies = []
                self.critic_losses = []

    def train_a_live(self):
        obs = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            states = []
            values = []
            actions = []
            rewards = []
            dones = []

            cur_steps = 0
            while cur_steps < self.n and not done:
                states.append(obs)
                action, value = self.action_value(obs[None, :])
                values.append(value)
                actions.append(action)
                self.queue.put(1)
                # self.env.render()
                obs, reward, done, info = self.env.step(action + 1)  # 3 or 4
                if done:
                    rewards.append(np.sign(reward))
                else:
                    rewards.append(np.sign(reward))
                dones.append(done)
                total_reward += reward
                cur_steps += 1

            _, next_value = self.action_value(obs[None, :])
            actions = np.array(actions)
            rewards = np.array(rewards)
            dones = np.array(dones)
            values = np.squeeze(np.array(values), axis=-1)
            returns, advs = self.advantages(rewards, dones, values, next_value)
            acts_advs = np.concatenate([actions[:, None], advs[:, None]], axis=-1)
            history = self.model.fit(np.array(states), [acts_advs, acts_advs, returns],
                                     verbose=0, callbacks=[self.cb, self.cp])

            actor_loss = history.history['output_1_loss'][0]
            entropy = history.history['output_2_loss'][0]
            critic_loss = history.history['output_3_loss'][0]
            self.actor_losses.append(actor_loss)
            self.entropies.append(entropy)
            self.critic_losses.append(critic_loss)
        return total_reward

    def action_value(self, obs):
        policy, _, value = self.model.predict_on_batch(obs)
        action = self.get_action(policy)
        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)

    def get_action(self, policy):
        return tf.squeeze(tf.random.categorical(policy, 1), axis=-1)

    def advantages(self, rewards, dones, values, next_value):
        returns = np.append(np.zeros_like(rewards), next_value, axis=-1)
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def stop(self):
        self.running = False

