from tensorflow.python.ops import rnn_cell, seq2seq
from tensorflow.python.ops.seq2seq import linear
from .seq_seq import SeqToSeq
import tensorflow as tf


class TFSeq2Seq(SeqToSeq):
    def __init__(self, args, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)
        self.args = args

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.rnn_size, state_is_tuple=False)

        self.cell = cell = rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=False)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.intention_targets = tf.placeholder(tf.int32, [args.batch_size, 1])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
            inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = linear(output, output_size=args.tag_count)
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.targets, [-1])],
                                                [tf.sign(tf.reshape(self.targets, [-1]))])
        self.final_state = last_state

        self.intentions = linear(last_state, output_size=args.act_count)
        self.intentions = tf.sigmoid(self.intentions)
        self.intention_probs = tf.nn.softmax(self.intentions)

        loss2 = seq2seq.sequence_loss_by_example([self.intentions],
                                                 [tf.reshape(self.intention_targets, [-1])],
                                                 [tf.ones(args.batch_size)])

        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length \
                    + loss2 / args.batch_size

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.check = tf.add_check_numerics_ops()
