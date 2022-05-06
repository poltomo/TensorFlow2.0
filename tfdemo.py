import tensorflow as tf

# vector is a one dimensional tensor
x=tf.constant(5)
print(x)

# matrix is a two dimensional tensor
x=tf.constant([[1,2,3],[4,5,6]])
print(x)

x=tf.random.normal((3,3), mean=0, stddev=1)
print(x)
