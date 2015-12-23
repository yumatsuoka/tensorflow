import tensorflow as tf
import input_data

#get mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#10 class distribution so output is 10 dimention

#input_data. a box to put input  data 
x = tf.placeholder(tf.float32, [None, 784])
#weight. output is 10 dim, input is 784 dim, 
W = tf.Variable(tf.zeros([784, 10]))
#bias.  W*x + b so 10 dim
b = tf.Variable(tf.zeros([10]))
#implement. tf.matmul is 内積. y is output
y = tf.nn.softmax(tf.matmul(x, W) + b)
#a box to put a targets
y_ = tf.placeholder(tf.float32, [None, 10])
#reduce_sum is summary of tensor. 
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#to minimize stuf using  cross entropy. a learning rate is 0.01 
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#initialize the variables we created
init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for i in range(1000):
    #size of batch is 100 data.
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #train_step is stuff to replace the placeholders
    #if implement 1 batch, apply minimize.
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#出力層の中で一番大きく出力している部分をargmaxで返す。
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels})
