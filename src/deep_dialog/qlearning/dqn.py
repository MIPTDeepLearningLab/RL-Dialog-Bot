"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461

Based on: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import pickle

np.random.seed(1)
tf.set_random_seed(1)

class DoubleDQN:
    def __init__(self,**kwargs):
        self.kwargs = kwargs

        self.hidden_size = kwargs.get('hidden_size', 80)
        self.n_actions = kwargs.get('n_actions')
        self.n_features = kwargs.get('n_features')
        self.lr = kwargs.get('learning_rate', 0.001)
        self.gamma = kwargs.get('reward_decay', 0.999)
        self.epsilon_max = kwargs.get('e_greedy', 0.9)
        self.batch_size = kwargs.get('batch_size', 16)
        self.epsilon_increment = kwargs.get('e_greedy_increment')
        #self.epsilon = 0 if kwargs.get('e_greedy_increment', False) else self.epsilon_max
        self.epsilon = 1

        self.double_q = kwargs.get('double_q', True)    # decide to use double q or not

        sess = kwargs.get('sess')
        self._build_net()
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if kwargs.get('output_graph'):
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.hidden_size, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def choose_action(self, observation):
        if len(observation.shape)!=2:
            observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.random.randint(0, self.n_actions)
        return action

    def replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

        print('\ntarget_params_replaced\n')

    def learn(self, batch_memory):
        batch_memory = np.array(batch_memory)
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        return {'cost': {'total_cost': self.cost}}

    def save(self, filepath):
        self.kwargs['sess'] = None
        pickle.dump(self.kwargs, open(filepath + '.params', 'wb'))

        saver = tf.train.Saver(max_to_keep=1)
        # Save model weights to disk
        save_path= saver.save(self.sess, filepath, global_step=0)
        print('Save path = {}'.format(save_path))

    @classmethod
    def load(self, filepath):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        constructor_params = pickle.load(open(filepath.split('.tf')[0]+'.tf.params', 'rb'))
        constructor_params['sess'] = sess
        result = DoubleDQN(**constructor_params)
        saver = tf.train.Saver()
        sess = saver.restore(sess, filepath)
        return result

