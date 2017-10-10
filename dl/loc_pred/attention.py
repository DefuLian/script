from keras.layers import Layer
import keras.backend as K
class Attention(Layer):
    def __init__(self, **kwargs): # [Batch][Dim1]
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1]
            self.Wq = self.add_weight(name='kernal_query_features', shape=(self.query_dim, self.att_size), initializer='glorot_normal', trainable=True)
        else:
            self.att_size = input_shape[-1]

        self.Wh = self.add_weight(name='kernal_hidden_features', shape=(self.att_size,self.att_size), initializer='glorot_normal', trainable=True)
        self.v = self.add_weight(name='query_vector', shape=(self.att_size, 1), initializer='zeros', trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        '''
        :param inputs: a list of tensor of length not larger than 2, or a memory tensor of size BxTXD1.
        If a list, the first entry is memory, and the second one is query tensor of size BxD2 if any
        :param mask: the masking entry will be directly discarded
        :return: a tensor of size BxD1, weighted summing along the sequence dimension
        '''
        if isinstance(inputs, list) and len(inputs) == 2:
            memory, query = inputs
            hidden = K.dot(memory, self.Wh) + K.expand_dims(K.dot(query, self.Wq), 1)
            mask = mask[0]
        else:
            if isinstance(inputs, list):
                if len(inputs) != 1:
                    raise ValueError('inputs length should not be larger than 2')
                memory = inputs[0]
            else:
                memory = inputs
            hidden = K.dot(memory, self.Wh)

        hidden = K.tanh(hidden)
        s = K.softmax(K.squeeze(K.dot(hidden, self.v), -1))
        if mask is not None:
            s *= K.cast(mask, dtype='float32')
        sum_by_time = K.sum(s, axis=-1, keepdims=True)
        s = s / (sum_by_time + K.epsilon())
        return K.sum(memory * K.expand_dims(s), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            att_size = input_shape[0][-1]
            batch = input_shape[0][0]
        else:
            att_size = input_shape[-1]
            batch = input_shape[0]
        return (batch, att_size)