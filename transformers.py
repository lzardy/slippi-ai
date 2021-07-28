import sonnet as snt
import tensorflow as tf
import math

def positional_encoding(seq_len, d_model, batch_size=1):
    """
    Returns a tensor following the postional encoding function
     (sinusodal from Vaswani et. all 2017).

    Return shape: [batch, seq_len, d_model]
    """
    def encoding_angle(pos, i):
        pos = tf.cast(pos, tf.dtypes.float32)
        i = tf.cast(i, tf.dtypes.float32)
        denom = tf.map_fn(fn=lambda t: tf.math.pow(10000, t//d_model), elems=i)
        return pos / denom

    i_tensor = tf.expand_dims(tf.range(0, d_model), 0) # [1, d_model]
    i_tensor = tf.repeat(i_tensor, [seq_len], axis=0) # [seq_len, d_model]
    j_tensor = tf.expand_dims(tf.range(0, seq_len), 1)
    j_tensor = tf.broadcast_to(j_tensor, [seq_len, d_model])
    #pos_tensor = tf.stack(i_tensor, j_tensor)
    # Double check these values for floating point error
    angles = encoding_angle(j_tensor, i_tensor)

    # Apply sin to even indices, cos to odd indices
    # TODO there's probably a fancy way to do this in place & save lots of memory
    evens = tf.math.sin(angles[:, 0::2])
    evens = tf.expand_dims(evens, -1) # [s, d, 1]
    odds = tf.math.cos(angles[:, 1::2])
    odds = tf.expand_dims(odds, -1) # [s, d, 1]
    joined = tf.concat([odds, evens], -1) # [s, d, 2]
    encoding = tf.reshape(joined, [seq_len, d_model]) # [s, d]

    #Add in batch
    encoding = tf.expand_dims(encoding, 0)
    encoding = tf.repeat(encoding, [batch_size], axis=0) # [b, s, d]
    return encoding

def attention(queries, keys, values):
    """
    Returns the 'attention' between three sequences: keys, queries, and values
    Specifically this implementation uses 'scaled dot-product' attention.

    This can be seen as a measure of the compatibility or relative importance between the keys and queries.
    This compatilbility is then applied to the 'input' sequence represented by values.

    Returns a tensor with the same shape as Values where [b, i ,j] represents the relative importance
    "attention" of element j in the sequence.

    keys: (batch, seq_len, D_k)
    queries: (batch, seq_len, D_k)
    values: (batch, seq_len, D_v)
    returns: (batch, seq_len, D_v)
    """
    assert keys.shape == queries.shape, "keys and values must have equivalent shapes"
    # compat [b, i, j] is the dot product of key i and query j (for batch # b)
    compat = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, S, S]
    # TODO This needs to be masked along the diagnol
    norm_compat = compat / math.sqrt(keys.shape[-1]) # [B, S, S]
    probs = tf.nn.softmax(norm_compat) # [B, S, S]
    att = tf.matmul(probs, values) # [B, S, D_V]
    return att

class MultiHeadAttentionBlock(snt.Module):
  def __init__(self, num_heads, output_size, name='MultiHeadAttentionBlock'):
    super(MultiHeadAttentionBlock, self).__init__()
    self.num_heads = num_heads
    self.W_K = []
    self.W_V = []
    self.W_Q = []
    assert output_size%num_heads == 0, "output_size must be a multiple of num_heads"
    projection_size = output_size/num_heads
    for _ in range(num_heads):
        # TODO there's a more efficient way to do this
      self.W_K.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_V.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
      self.W_Q.append(snt.Linear(int(projection_size))) #output is d_model/num_heads
    self.W_O = snt.Linear(output_size)

  def initial_state(self, batch_size):
    raise NotImplementedError()

  def __call__(self, inputs):
    """
    For each head, this block will project input into 3 spaces (keys, queries, values)
    and subsequently run an attention block on each projection. The results of each heads are
    combined (via concat) into the final output.

    inputs: [B, S, D_m]
    returns: [B, S, D_m]
    """
    # MHA(Q, K, V) = Concat(head_1...head_h)W^O
    heads = []
    for i in range(self.num_heads):
      # head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
      head_i = attention(self.W_Q[i](inputs), self.W_K[i](inputs), self.W_V[i](inputs)) # [B, S, D_m/h]
      heads.append(head_i)
    multi_head = self.W_O(tf.concat(heads, -1))  # [B, S, D_m]
    return multi_head

class TransformerEncoderBlock(snt.Module):
    def __init__(self, name: "EncoderTransformer"):
        super(TransformerEncoderBlock, self).__init__()

    def initial_state(self, batch_size):
        raise NotImplementedError()

    def __call__(self, inputs):
        raise NotImplementedError()
        # 1 positional encoding
        # N encoder blocks
            # MHA
            # Add/Norm (w/ residual from inputs)
            # Feed forward
            # Add/Norm (w/ residual from output of prev Add/Norm)