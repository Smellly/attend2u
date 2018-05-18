from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import inspect
import tensorflow as tf
import os

from scripts.generate_dataset import GO_ID

sequence_loss = tf.contrib.legacy_seq2seq.sequence_loss
layers = tf.contrib.layers
arg_scope = tf.contrib.framework.arg_scope

class CSMN(object):
  """Context Sequence Memory Network"""
  def _embedding_to_hidden(self, input2dim, size, scope='Wh', reuse = True):
    with arg_scope([layers.fully_connected],
            num_outputs = self.mem_dim,
            activation_fn = tf.nn.relu,
            weights_initializer = self.w_initializer,
            biases_initializer = self.b_initializer
    ):
      output2dim = tf.reshape(
          layers.fully_connected(
              input2dim,
              reuse = reuse,
              scope=scope
          ),
          [-1, size, self.mem_dim]
      )
    return output2dim

  def _init_img_mem(self, conv_cnn):
    """
    Returns:
        words_mem_A: [batch_size, img_memory_size, mem_dim]
        words_mem_C: [batch_size, img_memory_size, mem_dim]
    """
    img_mem_A = self._embedding_to_hidden(
        conv_cnn,
        1,
        scope = 'Wima',
        reuse = False
    )
    img_mem_C = self._embedding_to_hidden(
        conv_cnn,
        1,
        scope = 'Wimc',
        reuse = False
    )
    return img_mem_A, img_mem_C
  
  def _text_cnn(self, inputs, filter_sizes, mem_size, scope):
    """Text CNN. Code from https://github.com/dennybritz/cnn-text-classification-tf

    Args:
        input:
        weights:
        biases:
        filter_sizes:
        mem_size:
    Returns:
        pooled_outputs:
    """
    pooled_outputs = []
    with arg_scope([layers.conv2d],
          stride=1,
          padding='VALID',
          activation_fn = tf.nn.relu,
          weights_initializer = self.w_initializer,
          biases_initializer = self.b_initializer
    ):
      for j, filter_size in enumerate(filter_sizes):
        conv = layers.conv2d(
            inputs,
            self.num_channels,
            [filter_size, self.mem_dim],
            scope=scope + '-conv%s' % filter_size
        )
        pooled = layers.max_pool2d(
            conv,
            [mem_size- filter_size + 1, 1],
            stride= 1,
            padding='VALID',
            scope=scope + '-pool%s' % filter_size
        )
        pooled_outputs.append(pooled)
    return pooled_outputs

  def _init_words_mem(self, words, shape, words_mask, max_length, is_first_time, is_init_B):
    """
    Returns:
        words_mem_A: [batch_size, max_length, mem_dim]
        words_mem_B: [batch_size, max_length, mem_dim]
        words_mem_C: [batch_size, max_length, mem_dim]
    """
    shape_pad_size = tf.stack([1, max_length - shape[1], 1])
    padding = tf.tile(
        tf.constant(
            0.0,
            shape=[self.batch_size, 1, self.word_emb_dim]
        ),
        shape_pad_size
    )

    emb_A = tf.nn.embedding_lookup(self.Wea, tf.slice(words, [0, 0], shape))
    emb_C = tf.nn.embedding_lookup(self.Wec, tf.slice(words, [0, 0], shape))
    padded_emb_A = tf.concat([emb_A, padding], 1)
    padded_emb_C = tf.concat([emb_C, padding], 1)

    tiled_words_mask = tf.tile(
        tf.reshape(
            words_mask,
            [self.batch_size, max_length, 1]
        ),
        [1, 1, self.mem_dim]
    )

    embedding_shape = [self.batch_size, max_length, self.word_emb_dim]

    words_mem_A = self._embedding_to_hidden(
        padded_emb_A, embedding_shape[1],
        reuse = not is_first_time
    ) * tf.to_float(tiled_words_mask)
    words_mem_C = self._embedding_to_hidden(
        padded_emb_C,
        embedding_shape[1]
    ) * tf.to_float(tiled_words_mask)
    if is_init_B:
      emb_B = tf.nn.embedding_lookup(self.Web, tf.slice(words, [0, 0], shape))
      padded_emb_B = tf.concat([emb_B, padding], 1)
      words_mem_B = self._embedding_to_hidden(
          padded_emb_B,
          embedding_shape[1],
      ) * tf.to_float(tiled_words_mask)
      return words_mem_A, words_mem_B, words_mem_C
    return words_mem_A, words_mem_C

  def __init__(self, inputs, config, name="CSMN", is_training=True):
    def _init_tpc_mem():
      with open(os.path.join('data', 'Topic_list.txt'), 'r') as f:
          topic_list = f.readlines()
      topic_word = [x.split()[0] for x in topic_list]
      with open(os.path.join('data', 'caption_dataset', '40000.vocab'), 'r') as f:
          vocab = f.read().split('\n')
      topic_ind = []
      for x in topic_word:
          if x in vocab:
              topic_ind.append(vocab.index(x))
          else:
              print('not int topic word:', x)

      u_matrix =  tf.get_variable(
              'u_matrix', 
              shape=[self.mem_dim, self.mem_dim],
              initializer=self.w_initializer,
              trainable=True
            )
      return tf.reshape(tf.convert_to_tensor(topic_ind, dtype=tf.int32), [-1, 1]), u_matrix

    (conv_cnn, context_largest_length, output_largest_length, topic_largest_length, 
        context, caption, answer, topic, context_mask, output_mask) = inputs
    context_largest_length = tf.reduce_max(context_largest_length)
    output_largest_length = tf.reduce_max(output_largest_length)
    # Set config's variables as models' variables
    attrs = {
        k: v for k, v in inspect.getmembers(config) \
            if not k.startswith('__') and not callable(k)
    }
    for attr in attrs:
      name = attr if not attr.startswith('_') else attr[1:]
      setattr(self, name, getattr(config, attr))

    self.is_training = is_training
    self.name = name
    self.w_initializer = tf.uniform_unit_scaling_initializer(1.0)
    # self.w_initializer = tf.initializers.variance_scaling(1.0)
    self.b_initializer = tf.constant_initializer(0.0)

    emb_shape = [self.vocab_size, self.word_emb_dim]

    self.Wea  = tf.get_variable(
        "Wea",
        shape=emb_shape,
        initializer=self.w_initializer
    )

    self.Web  = tf.get_variable(
        "Web",
        shape=emb_shape,
        initializer=self.w_initializer
    )

    self.Wec  = tf.get_variable(
        "Wec",
        shape=emb_shape,
        initializer=self.w_initializer
    )
    self.Wf = tf.get_variable(
        "Wf",
        shape=[self.num_channels_total, self.vocab_size],
        initializer=self.w_initializer
    )
    self.bf = tf.get_variable(
        "bf",
        shape=[self.vocab_size],
        initializer=self.b_initializer
    )

    # Build Memories
    # Image Memory
    img_mem_A, img_mem_C = self._init_img_mem(conv_cnn)
    # User Context Memory
    # context_mem_A, context_mem_C = self._init_words_mem(
    _, _ = self._init_words_mem(
        context,
        tf.stack([self.batch_size, context_largest_length]),
        context_mask,
        self.max_context_length,
        is_first_time = True,
        is_init_B = False
    )
    topic_ind, u_matrix = _init_tpc_mem()

    # Word Output Memory
    if self.is_training:
      output_mem_A, output_mem_B, output_mem_C = self._init_words_mem(
          caption,
          tf.stack([self.batch_size, output_largest_length]),
          output_mask,
          self.max_output_length,
          is_first_time = False,
          is_init_B = True
      )
    else:
      output_mem_A, output_mem_B, output_mem_C = \
          self.Wea, self.Web, self.Wec


    def _loop_cond(iterator_, output_words_array_,
                   output_mem_state_A_, output_mem_state_C_,
                   output_mem_A_, output_mem_B_, output_mem_C_):
      if self.is_training:
        return tf.less(iterator_, output_largest_length)
      else:
        return tf.less(iterator_, self.max_output_length)

    def _loop_body(iterator_, output_words_array_,
					output_mem_state_A_, output_mem_state_C_,
					output_mem_A_, output_mem_B_, output_mem_C_):
      def _train_input():
        output_A_slice = tf.slice(
            output_mem_A_,
            [0, iterator_, 0],
            [self.batch_size, 1, self.mem_dim]
        )
        output_C_slice = tf.slice(
            output_mem_C_,
            [0, iterator_, 0],
            [self.batch_size, 1, self.mem_dim]
        )
        query = tf.slice(
            output_mem_B_,
            [0, iterator_, 0],
            [self.batch_size, 1, self.mem_dim]
        )
        return output_A_slice, output_C_slice, query

      def _test_input():
        def _go_symbol():
            # GO_ID = 1
          return tf.constant(GO_ID, shape=[self.batch_size, 1], dtype=tf.int32)

        def _not_go_symbol():
          out = output_words_array_.read(iterator_ - 1)
          out.set_shape([self.batch_size, self.num_channels_total])
          out = tf.matmul(out, self.Wf) + self.bf
          out = tf.reshape(
              tf.to_int32(tf.argmax(tf.nn.softmax(out), 1)),
              [self.batch_size, 1]
          )
          return out
      
        prev_word = tf.cond(tf.equal(iterator_, 0), _go_symbol, _not_go_symbol)

        output_A_slice = self._embedding_to_hidden(
            tf.nn.embedding_lookup(output_mem_A_, prev_word),
            1
        )
        output_C_slice = self._embedding_to_hidden(
            tf.nn.embedding_lookup(output_mem_C_, prev_word),
            1
        )
        query = self._embedding_to_hidden(
            tf.nn.embedding_lookup(output_mem_B_, prev_word),
            1
        )

        return output_A_slice, output_C_slice, query

      # Build current slice
      output_A_slice, output_C_slice, query = _train_input() \
          if self.is_training else _test_input()

      # Build query
      query = self._embedding_to_hidden(
          query,
          1,
          scope = "Wq",
          reuse = False
      )

      # Update memory network by appending new word
      output_Ai_slice = tf.slice(
          output_mem_state_A_,
          [0, 0, 0],
          [self.batch_size, iterator_, self.mem_dim]
      )
      output_Ci_slice = tf.slice(
          output_mem_state_C_,
          [0, 0, 0],
          [self.batch_size, iterator_, self.mem_dim]
      )
      output_padding = tf.tile(
          tf.constant(0.0, shape=[self.batch_size, 1, self.mem_dim]),
          [1, self.max_output_length - iterator_ - 1, 1]
      )

      output_mem_state_A_ = tf.concat(
          [output_Ai_slice, output_A_slice, output_padding],
          1
      )
      output_mem_state_C_ = tf.concat(
          [output_Ci_slice, output_C_slice, output_padding],
          1
      )
      # Need to specify shape in while loop
      output_mem_state_A_ = tf.reshape(
          output_mem_state_A_,
          [self.batch_size, self.max_output_length, self.mem_dim]
      )
      output_mem_state_C_ = tf.reshape(
          output_mem_state_C_,
          [self.batch_size, self.max_output_length, self.mem_dim]
      )

      # Memory network computation (mem_A -> attention -> mem_Catt)
      # mem_A = tf.concat([img_mem_A, context_mem_A, output_mem_state_A_], 1)
      mem_A = tf.concat([img_mem_A, output_mem_state_A_], 1)
      # innerp_mem_A = tf.matmul(query, mem_A, adjoint_b=True)

      memory_sizes = [
          self.img_memory_size,
          # self.max_context_length,
          self.max_output_length
      ]

      # print('shape of output_mem_state_A:', output_mem_state_A.get_shape())
      # print('shape of output_mem_B_:', output_mem_B_.get_shape())

      topic_emb = tf.nn.embedding_lookup(
          self.Web, 
          tf.slice(topic_ind, [0, 0], topic_ind.shape))

      topic_mem = tf.get_variable(
                  'topic_mem',
                  shape=[topic_ind.get_shape()[0], self.mem_dim],
                  trainable = False
              )
      topic_mem = self._embedding_to_hidden(
                  topic_emb, 
                  int(topic_ind.shape[0]),
                  reuse = True
                ) 
      topic_mem = tf.reshape(topic_mem, [self.mem_dim, -1])

      # print('shape of topic:', topic_ind.shape)
      # print('shape of img_mem_A:', img_mem_A.shape)
      # print('shape of output_mem_state_A:', output_mem_state_A_.shape)
      # print('shape of u_matrix:', u_matrix.shape)
      # print('shape of query:', query.shape)
      # print('shape of mem_A:', mem_A.shape)
      # print('shape of topic_mem:', topic_mem.shape)
      # print('shape of tf.matmul(u_matrix, mem_A):', tf.einsum('abi,ij->abj', mem_A, u_matrix).shape)

      alpha_m = tf.einsum(
                'abi,ij->abj', 
                tf.einsum('abi,ij->abj', mem_A, u_matrix), 
                topic_mem
            )
      alpha_m_re = tf.reduce_sum(alpha_m, axis=1)
      # print('shape of alpha_m_re:', alpha_m_re.shape)

      alpha_numerator = [ [] for i in range(self.batch_size) ]
      for i in range(self.batch_size):
          for j in range(self.max_topic_length):
              alpha_numerator[i].append(
                      tf.nn.embedding_lookup(alpha_m_re[i,:], topic[i, j]))
      alpha_numerator = tf.stack(alpha_numerator)
      # print('shape of alpha_numerator:', alpha_numerator.shape)
      alpha_numerator = tf.reduce_sum(
                            tf.exp(
                            tf.reduce_sum(
                                tf.reshape(
                                    alpha_numerator, 
                                    [self.batch_size, -1, self.max_topic_length]),
                                axis=1)),
                            axis=1)
      alpha_numerator = tf.reshape(alpha_numerator, [self.batch_size, -1])
      # print('shape of alpha_numerator:', alpha_numerator.shape)
      # stb = tf.constant(1e-8, tf.float32) * tf.ones([self.batch_size, 1])
      alpha_denomenator = tf.reshape(
                  tf.reduce_sum(tf.exp(alpha_m_re), axis = 1), 
                  [self.batch_size, -1]
              )
      # print('shape of alpha_denomenator after reshape:', alpha_denomenator.shape)
      alpha = tf.divide(alpha_numerator, alpha_denomenator)

      # print('shape of innerp_mem_A without alpha:', tf.matmul(query, mem_A, adjoint_b=True).shape)
      # print('shape of alpha:', alpha.shape)
      # innerp_mem_A = tf.reshape(tf.matmul(query, mem_A, adjoint_b=True), [self.batch_size, -1])
      innerp_mem_A = tf.multiply(
                  alpha, 
                  tf.reshape(
                      tf.matmul(query, mem_A, adjoint_b=True), 
                      [self.batch_size, -1]
                  )
              )

      # print('shape of innerp_mem_A:', innerp_mem_A.shape)
      # print('shape of img_memory_size:', self.img_memory_size)
      # print('shape of self.max_context_length:', self.max_context_length)
      # print('shape of self.max_output_length:', self.max_output_length)
      # print('shape of memory_size:', self.memory_size)

      attention = tf.nn.softmax(
          tf.reshape(innerp_mem_A, [-1, self.memory_size]),
          name='attention'
      )

      # img_attention, context_attention, output_attention = \
      img_attention, output_attention = \
          tf.split(attention, memory_sizes, axis=1)

      img_attention = tf.tile(
          tf.reshape(
              img_attention,
              [self.batch_size, self.img_memory_size, 1]
          ),
          [1, 1, self.mem_dim]
      )
      '''
      context_attention = tf.tile(
          tf.reshape(
              context_attention,
              [self.batch_size, self.max_context_length, 1]
          ),
          [1, 1, self.mem_dim]
      )
      '''
      output_attention = tf.tile(
          tf.reshape(
              output_attention,
              [self.batch_size, self.max_output_length, 1]
          ),
          [1, 1, self.mem_dim]
      )
      #pool5
      img_weighted_mem_C = tf.reshape(
          img_mem_C * img_attention,
          [self.batch_size, self.mem_dim]
      )
      '''
      context_weighted_mem_C = tf.expand_dims(
          context_mem_C * context_attention,
          -1
      )
      '''
      output_weighted_mem_C = tf.expand_dims(
          output_mem_state_C_ * output_attention,
          -1
      )

      # Memory CNN
      pooled_outputs = []
      '''
      pooled_outputs += self._text_cnn(
          context_weighted_mem_C,
          self.context_filter_sizes,
          self.max_context_length,
          scope = "context"
      )
      '''
      pooled_outputs += self._text_cnn(
          output_weighted_mem_C,
          self.output_filter_sizes,
          self.max_output_length,
          scope = "output"
      )

      # print('shape of img_attention:', img_attention.get_shape())
      # print('shape of output_attention:', output_attention.get_shape())
      # print('shape of attention:', attention.get_shape())
      # print('shape of output_mem_C:', output_mem_C.get_shape())
      # print('shape of output_weighted_mem_C:', output_weighted_mem_C.get_shape())
      h_pool = tf.concat(pooled_outputs, 3)
      h_pool_flat = tf.reshape(h_pool, [-1, self.num_channels_total])
      # print('shape of h_pool:', h_pool)
      # print('shape of h_pool_flat:', h_pool_flat)
      with arg_scope([layers.fully_connected],
              num_outputs = self.num_channels_total,
              activation_fn = tf.nn.relu,
              weights_initializer = self.w_initializer,
              biases_initializer = self.b_initializer
      ):
        img_feature = layers.fully_connected(
                img_weighted_mem_C,
                reuse = False,
                scope="Wpool5"
        )

        # print('shape of img_mem_C:', img_mem_C.get_shape())
        # print('shape of img_weighted_mem_C:', img_weighted_mem_C.get_shape())
        # print('shape of img_feature:', img_feature.get_shape())
        img_added_result = img_feature +  h_pool_flat
        # print('shape of img_added_result:', img_added_result.get_shape())
        output = layers.fully_connected(
                img_added_result,
                reuse = False,
                scope="Wo"
        )

      output_words_array_ = output_words_array_.write(iterator_, output)

      return (iterator_ + 1, output_words_array_, output_mem_state_A_,
          output_mem_state_C_, output_mem_A_, output_mem_B_, output_mem_C_)

    iterator = tf.constant(0, dtype=tf.int32)
    output_words_array = tf.TensorArray(
        dtype=tf.float32,
        size=0,
        clear_after_read=False,
        dynamic_size=True
    )

    #initialize output memory state
    output_mem_state_A = tf.constant(
        0,
        dtype=tf.float32,
        shape=[self.batch_size, self.max_output_length, self.mem_dim]
    )
    output_mem_state_C = tf.constant(
        0,
        dtype=tf.float32,
        shape=[self.batch_size, self.max_output_length, self.mem_dim]
    )
    loop_vars = [
        iterator,
        output_words_array,
        output_mem_state_A,
        output_mem_state_C,
        output_mem_A,
        output_mem_B,
        output_mem_C
    ]
    #Don't make parallel
    loop_outputs = tf.while_loop(
        _loop_cond,
        _loop_body,
        loop_vars,
        back_prop=True,
        parallel_iterations=1
    )
    sequence_outputs = loop_outputs[1].stack()
    sequence_outputs.set_shape(
        [None, self.batch_size, self.num_channels_total]
    )
    self.om_s_a = loop_outputs[2]
    sequence_outputs = tf.transpose(sequence_outputs, perm=[1, 0, 2])
    sequence_outputs = tf.reshape(sequence_outputs, [-1, self.num_channels_total])
    final_outputs = tf.matmul(sequence_outputs, self.Wf) + self.bf
    prob = tf.nn.softmax(final_outputs)
    self.prob = tf.reshape(prob, [self.batch_size, -1, self.vocab_size])
    self.argmax = tf.argmax(self.prob, 2)

    output_mask = tf.slice(
        output_mask,
        [0, 0],
        [self.batch_size, output_largest_length]
    )
    answer = tf.slice(answer, [0, 0], [self.batch_size, output_largest_length])

    # print('sequence_outputs:', sequence_outputs.get_shape())
    # print('self.prob:', self.prob)
    # print('loop_outputs:', loop_outputs)
    # print('output_mask:', output_mask)
    # print('answer:', answer)
    self.loss = sequence_loss(
        [final_outputs],
        [tf.reshape(answer, [-1])],
        [tf.reshape(tf.to_float(output_mask), [-1])]
    )
    output_words_array.close()


