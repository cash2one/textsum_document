# Problem solve

## initial embedding problem 
Please check the code in [here](https://github.com/tensorflow/models/blob/master/textsum/seq2seq_attention_model.py) in class Seq2seqAttentionModel and function _add_seq2seq(self). As for one simple example of function tf.stop_gradient(), please refer to problem_solve/example.py

    # Embedding shared by the input and outputs.
    with tf.variable_scope('embedding'), tf.device('/cpu:0'):
    embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
    emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in encoder_inputs]
    emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in decoder_inputs]


    # Embedding shared by the input and outputs.
    with tf.variable_scope('embedding'), tf.device('/cpu:0'):
    #embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
    init = np.array(A) # comment suppose the embedding vector is A
    embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,initializer=tf.constant_initializer(init))
    embedding = tf.stop_gradient(embedding)
    emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in encoder_inputs]
    emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in decoder_inputs]


## attention mechanism
The code is [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py). The mechanism in in function attention_decoder. Check the code below: s is e_ij, a is a_ij, d is c_i (please check the comment below and check the paper in [here](https://arxiv.org/pdf/1409.0473v7.pdf)
    
    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(1, query_list)
      for a in xrange(num_heads): # here a is just a loop. I do not think it is good, since it conflicts another a below
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(
              v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3]) # e_ij in paper https://arxiv.org/pdf/1409.0473v7.pdf
          a = nn_ops.softmax(s) # a_ij in paper https://arxiv.org/pdf/1409.0473v7.pdf
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
              [1, 2]) # c_i in paper https://arxiv.org/pdf/1409.0473v7.pdf
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds

