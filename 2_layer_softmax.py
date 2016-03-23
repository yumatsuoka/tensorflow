# -*-coding:utf-8-*-
#2layer softmax neural network

#like python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#InteractiveSessionはグラフを作るときに柔軟に役立つよう！
import tensorflow as tf
sess = tf.InteractiveSession()

#get mnist data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#input_data, shape=[None is batch size, 784 is inputdata size]
x  = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#variable 変数, 784入力を１０出力にしたい, 10クラス分類だから
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#微小な値で変数(Variable)の初期値を埋めてくれる？のかな
sess.run(tf.initialize_all_variables())

#重みと入力の内積をとって、ベクトルを足し合わせてソフトマックス確率を計算する
y = tf.nn.softmax(tf.matmul(x, W) + b)

#誤差を算出する。クロスエントロピーを使う。バッチ単位で誤差を計算したほうが全体的に良い
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#学習をどういう感じでやってくかを決める
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    #placeholderにfeed_dictを使って値を入れていく
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

#tf.argmaxは１つの軸の中でもっとも高いインデックスを返す。その値が等しいかどうかtf.equalで確かめる
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#booleanのリストを返して、要素の割合をfloat型の値で返す。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))








