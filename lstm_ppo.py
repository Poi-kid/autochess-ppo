import numpy as np
import tensorflow as tf

class PPO:
    def __init__(self, s_dim, a_dim, target=None, clip_epsilon=0.2, critic_learning_rate=0.01, actor_learning_rate=0.02,
                 actor_update_steps=10, critic_update_steps=10):
        self.sess = tf.Session(target=target)

        self.s_dim = s_dim
        self.a_dim = a_dim
        self.clip_epsilon = clip_epsilon
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.actor_update_steps = actor_update_steps
        self.critic_update_steps = critic_update_steps

        # 定义state，action，reward
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.a = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.adv = tf.placeholder(tf.float32, [None, 1], 'reward')

        # 定义两个actor（old_actor固定住）
        self.actor, self.actor_params = self._build_actor_net('actor', trainable=True)
        self.old_actor, self.old_actor_params = self._build_actor_net('old_actor', trainable=False)

        # 计算ppo2的loss
        with tf.variable_scope('loss'):
            # 似然函数
            ratio = self.actor.prob(self.a) / self.old_actor.prob(self.a)
            likelihoods = ratio * self.adv
            # ppo2的loss
            self.loss = -tf.reduce_mean(tf.minimum(likelihoods, tf.clip_by_value(ratio, 1.-clip_epsilon, 1.+clip_epsilon) * self.adv))

        # 定义actor训练器
        self.actor_train_op = tf.train.AdamOptimizer(actor_learning_rate).minimize(self.loss)

        # 定义critic（判断局面价值）
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.discounted_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.critic_adv = self.discounted_r - self.v
            self.critic_loss = tf.reduce_mean(tf.square(self.critic_adv))
            self.critic_train_op = tf.train.AdamOptimizer(critic_learning_rate).minimize(self.critic_loss)

        # 采样函数（即使用actor根据state输出action，采集一次样本）
        with tf.variable_scope('sample_data'):
            self.sample_op = tf.squeeze(self.actor.sample(1), axis=0)

        # 使用actor参数更新old_actor参数
        with tf.variable_scope('update_old_actor'):
            self.update_old_actor_op = [o_a_p.assign(a_p) for a_p, o_a_p in zip(self.actor_params, self.old_actor_params)]

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def _build_actor_net(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.s, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, self.s_dim, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, allow_nan_stats=True)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def update(self, s, a, r):
        self.sess.run(self.update_old_actor_op)

        adv = self.sess.run(self.critic_adv, {self.s: s, self.discounted_r: r})

        # 更新actor
        [self.sess.run(self.actor_train_op, {self.s: s, self.a: a, self.adv: adv}) for _ in range(self.actor_update_steps)]

        # 更新critic
        [self.sess.run(self.critic_train_op, {self.s: s, self.discounted_r: r}) for _ in range(self.critic_update_steps)]

    def load_model(self):

        self.saver.restore(self.sess, './model/lstm_ppo')

    def save_model(self):
        self.saver.save(self.sess, './model/lstm_ppo')

    def action(self, s):
        s = s[np.newaxis, :]
        prob_weights = self.sess.run(self.sample_op, {self.s: s})
        return prob_weights
