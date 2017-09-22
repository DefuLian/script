
import tensorflow as tf
from keras.layers import Dot, Input, Lambda, multiply, dot
import numpy as np
from keras import backend as K
sess = K.get_session()
with sess.as_default():
    x1 = K.constant(np.random.rand(1,2,3))
    x2 = K.constant(np.random.rand(1,2,3))
    y = dot([x1,x2], axes=2)
    #y = Lambda(lambda v: K.sum(v,axis=-2))(y)
    #print(x1.eval()); print( '\n'); print(x2.eval()); print('\n'); print( y.eval()); print(y.shape)

    dataset = tf.contrib.data.Dataset.range(5)
    dataset = dataset.shuffle(2)
    #dataset = dataset.batch(3)
    #dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
    #dataset = dataset.padded_batch(4, padded_shapes=[None])

    iterator = dataset.make_initializable_iterator()
    sess.run(iterator.initializer)
    next1 = iterator.get_next()
    while True:
        try:
            v1= sess.run(next1)
            print(v1)
        except tf.errors.OutOfRangeError:
            break



#print y.eval()
#print y.shape
