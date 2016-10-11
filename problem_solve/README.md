# Problem solve

## initial embedding problem 
Please check the code in [here](https://github.com/tensorflow/models/blob/master/textsum/seq2seq_attention_model.py) in class Seq2seqAttentionModel and function _add_seq2seq(self)

    # Embedding shared by the input and outputs.
    with tf.variable_scope('embedding'), tf.device('/cpu:0'):
    embedding = tf.get_variable('embedding', [vsize, hps.emb_dim], dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=1e-4))
    emb_encoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in encoder_inputs]
    emb_decoder_inputs = [tf.nn.embedding_lookup(embedding, x) for x in decoder_inputs]
