# -*-coding:utf-8-*-
#Multilayer Convolutional Network in tutorials of tensorflow 
#2015/12/29

#like python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#get mnist data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#InteractiveSessionはグラフを作るときに柔軟に役立つらしい！
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

#weight_initialization
#小さな大きさのノイズを重みとバイアスに加える。勾配が０になるの妨げるため。
#ReLUを使うから、誤差が出ない死んだニューロンを作らないためにこの関数を作る。
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#ストライドは１だから、入力と同じサイズの出力が出てくる
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#2x2のサイズのプーリングを行う。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#最初の２つはパッチのサイズ。つぎは入力のチャンネル数、最後はアウトプットのチャンネルの数
w_conv1 = weight_variable([5, 5, 1, 32])
#バイアスはアウトプットチャンネルと同じ大きさ
b_conv1 = bias_variable([32])
#レイヤーに適用するために、リサイズする。最初の１つは4d tensorにするための値
#２つめと３つめは画像の横と縦に対応している。最後のものは色の次元数に一致する。
#最初の-1は１次元にするためのもの？
x_image = tf.reshape(x, [-1, 28, 28, 1])

#入力画像と重みをcovolveしてバイアスを加えて、ReLU関数にぶち込む。
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
#活性化関数から出てきたものをmax_poolingする。
h_pool1 = max_pool_2x2(h_conv1)

#２つ目の層では５×５パッチで、６４の特徴を作るように設定
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely connected layer(全結合層)
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#2x2のマックスプーリングを２回行ったから、画像サイズは7x7になっている
#1024次元の全結合ニューロンを作る。リサイズしてベクトルに直す。
#ベクトルと重みの内積を取って、バイアスを加えてReLU関数に突っ込む。
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#overfitting(過学習)を防ぐためにドロップアウトを、リードアウトレイヤーの前に入れる。
#学習しているときには確率的にドロップアウトされるように、テストしているときはされないように
#placeholderを作る。tf.nn.dropoutを使えば自動的にマスクをかぶせるようにやってくれるよー
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#convoluteしてきたベクトルを回帰させるレイヤー(Readout Layer)に入れる。
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#softmax関数をかまして、確率の形で回帰を行う。
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

#学習をテスト
#誤差を算出する。クロスエントロピーを使う。バッチ単位で誤差を計算したほうが全体的に良い
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
#ADAM_optimizerは確率的勾配法よりもより洗練されている。
#keep_probはドロップアウトさせる確率で、feed_dictに加えている。
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#tf.argmaxは１つの軸の中でもっとも高いインデックスを返す。その値が等しいかどうかtf.equalで確かめる
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#booleanのリストを返して、要素の割合をfloat型の値で返す。
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#微小な値で変数(Variable)の初期値を埋めてくれる？のかな
sess.run(tf.initialize_all_variables())
#学習スタート
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" %(i, train_accuracy))
    #placeholderにfeed_dictを使って値を入れていく
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #debug
    #print(sess.run(cross_entropy, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})) 
    #

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

