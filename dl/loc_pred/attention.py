from keras.layers import Layer
import keras.backend as K
class Attention(Layer):
    def __init__(self, method=None, **kwargs):
        self.supports_masking = True
        if method != 'lba' and method !='ga' and method != 'cba' and method is not None:
            raise ValueError('attention method is not supported')
        self.method = method
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1]
            if self.method == 'ga' or self.method == 'cba':
                self.Wq = self.add_weight(name='kernal_query_features', shape=(self.query_dim, self.att_size), initializer='glorot_normal', trainable=True)
        else:
            self.att_size = input_shape[-1]

        if self.method == 'cba':
            self.Wh = self.add_weight(name='kernal_hidden_features', shape=(self.att_size,self.att_size), initializer='glorot_normal', trainable=True)
        if self.method == 'lba' or self.method == 'cba':
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
            if self.method is None:
                return memory[:,-1,:]
            elif self.method == 'cba':
                hidden = K.dot(memory, self.Wh) + K.expand_dims(K.dot(query, self.Wq), 1)
                hidden = K.tanh(hidden)
                s = K.squeeze(K.dot(hidden, self.v), -1)
            elif self.method == 'ga':
                s = K.sum(K.expand_dims(K.dot(query, self.Wq), 1) * memory, axis=-1)
            else:
                s = K.squeeze(K.dot(memory, self.v), -1)
            mask = mask[0]
        else:
            if isinstance(inputs, list):
                if len(inputs) != 1:
                    raise ValueError('inputs length should not be larger than 2')
                memory = inputs[0]
            else:
                memory = inputs
            if self.method is None:
                return memory[:,-1,:]
            elif self.method == 'cba':
                hidden = K.dot(memory, self.Wh)
                hidden = K.tanh(hidden)
                s = K.squeeze(K.dot(hidden, self.v), -1)
            elif self.method == 'ga':
                raise ValueError('general attention needs the second input')
            else:
                s = K.squeeze(K.dot(memory, self.v), -1)

        s = K.softmax(s)
        if mask is not None:
            s *= K.cast(mask, dtype='float32')

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
