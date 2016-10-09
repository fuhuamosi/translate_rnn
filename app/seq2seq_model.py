# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from app import data_utils
import numpy as np
import random

__author__ = 'fuhuamosi'


class Seq2SeqModel:
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False,
                 num_samples=512, forward_only=False):
        """
        Create the model.
            Args:
              source_vocab_size: size of the source vocabulary.
              target_vocab_size: size of the target vocabulary.
              buckets: a list of pairs (I, O), where I specifies maximum input length
                that will be processed in that bucket, and O specifies maximum output
                length. Training instances that have inputs longer than I or outputs
                longer than O will be pushed to the next bucket and padded accordingly.
                We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
              size: number of units in each layer of the model.
              num_layers: number of layers in the model.
              max_gradient_norm: gradients will be clipped to maximally this norm.
              batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
              learning_rate: learning rate to start with.
              learning_rate_decay_factor: decay learning rate by this much when needed.
              use_lstm: if true, we use LSTM cells instead of GRU cells.
              num_samples: number of samples for sampled softmax.
              forward_only: if set, we do not construct the backward pass in the model.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        output_projection = None
        softmax_loss_function = None
        if 0 < num_samples < target_vocab_size:
            w = tf.get_variable('proj_w', shape=(size, target_vocab_size))
            w_t = tf.transpose(w)
            b = tf.get_variable('proj_b', shape=(target_vocab_size,))
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, shape=(-1, 1))
                return tf.nn.sampled_softmax_loss(w_t, b, inputs, labels, num_samples,
                                                  self.target_vocab_size)

            softmax_loss_function = sampled_loss

        single_cell = tf.nn.rnn_cell.GRUCell(num_units=size)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.LSTMCell(num_units=size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.nn.seq2seq. \
                embedding_attention_seq2seq(encoder_inputs, decoder_inputs,
                                            cell,
                                            num_encoder_symbols=source_vocab_size,
                                            num_decoder_symbols=target_vocab_size,
                                            embedding_size=size,
                                            output_projection=output_projection,
                                            feed_previous=do_decode)

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(dtype=tf.int32, shape=[None],
                                                      name='encoder{}'.format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(dtype=tf.int32, shape=[None],
                                                      name='decoder{}'.format(i)))
            self.target_weights.append(tf.placeholder(dtype=tf.float32, shape=[None],
                                                      name='weight{}'.format(i)))
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        if forward_only:
            self.outputs, self.losses = tf.nn.seq2seq. \
                model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                   targets, self.target_weights, buckets,
                                   seq2seq=lambda x, y: seq2seq_f(x, y, True),
                                   softmax_loss_function=softmax_loss_function)
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [tf.matmul(output, output_projection[0]) + output_projection[1] for
                                       output in self.outputs[b]]
        else:
            self.outputs, self.losses = tf.nn.seq2seq. \
                model_with_buckets(self.encoder_inputs, self.decoder_inputs,
                                   targets, self.target_weights, buckets,
                                   seq2seq=lambda x, y: seq2seq_f(x, y, False),
                                   softmax_loss_function=softmax_loss_function)

        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(optimizer.apply_gradients(zip(clipped_gradients, params),
                                                              global_step=self.global_step))
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """
        Run a step of the model feeding the given inputs.
            Args:
              session: tensorflow session to use.
              encoder_inputs: list of numpy int vectors to feed as encoder inputs.
              decoder_inputs: list of numpy int vectors to feed as decoder inputs.
              target_weights: list of numpy float vectors to feed as target weights.
              bucket_id: which bucket of the model to use.
              forward_only: whether to do the backward step or only forward.
            Returns:
              A triple consisting of gradient norm (or None if we did not do backward),
              average perplexity, and the outputs.
            Raises:
              ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
            """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        input_feed = {}
        for i in range(encoder_size):
            input_feed[self.encoder_inputs[i].name] = encoder_inputs[i]
        for i in range(decoder_size):
            input_feed[self.decoder_inputs[i].name] = decoder_inputs[i]
            input_feed[self.target_weights[i].name] = target_weights[i]

        last_target = self.decoder_inputs[decoder_size]
        input_feed[last_target.name] = np.zeros(shape=[self.batch_size], dtype=np.int32)

        if not forward_only:
            output_fetch = [self.gradient_norms[bucket_id],
                            self.losses[bucket_id],
                            self.updates[bucket_id]]
        else:
            output_fetch = [self.losses[bucket_id]]
            for i in range(decoder_size):
                output_fetch.append(self.outputs[bucket_id][i])

        outputs = session.run(output_fetch, input_feed)
        if not forward_only:
            return outputs[0], outputs[1], None  # Gradient norm, loss, no output.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id):
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input) - 1)
            decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        for l in range(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch][l]
                                                  for batch in range(self.batch_size)],
                                                 dtype=np.int32))
        for l in range(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch][l]
                                                  for batch in range(self.batch_size)],
                                                 dtype=np.int32))
            batch_weight = np.ones([self.batch_size], dtype=np.float32)
            for batch in range(self.batch_size):
                if l == decoder_size - 1 \
                        or (l < decoder_size - 1
                            and batch_decoder_inputs[l][batch] == data_utils.PAD_ID):
                    batch_weight[batch] = 0
            batch_weights.append(batch_weight)
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
