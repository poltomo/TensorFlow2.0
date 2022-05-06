import tensorflow as tf

# vector is a one dimensional tensor
x=tf.constant(5)
print(x)

# matrix is a two dimensional tensor
x=tf.constant([[1,2,3],[4,5,6]])
print(x)

# tensorflow normal distribution
x=tf.random.normal((3,3), mean=0, stddev=1)
print(x)

# vector       0,1,2,3,4,5,6
x=tf.constant([1,2,3,4,5,9,8])

# tensor indexing
print(x)
print(x[::])
print(x[4:])
print(x[::-1])
print(x[::3])

x=tf.range(9)
print(x)

#vector -> matrix 3*3
y=tf.reshape(x,(3,3))