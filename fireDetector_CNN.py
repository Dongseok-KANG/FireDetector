import tensorflow as tf
import random
import glob
import numpy as np
import os.path as path
import cv2
import matplotlib.pyplot as plt

tf.set_random_seed(777)


# 파일 이름 가져오기
def loard_img(img_path):
    image_path = img_path
    file_path = glob.glob(path.join(image_path, '*.png'))

    # 파일 이름에 맞는 이미지 불러오기
    images = [cv2.imread(path, cv2.IMREAD_ANYCOLOR) for path in file_path]
    images = np.asarray(images, dtype=np.float32)

    # 이미지 값이 0~1이 되도록 이미지 크기를 변경
    images = images / 255

    # 이미지이름중 맨 앞 글자를 가져와서 라벨을 만든다
    n_images = images.shape[0]
    labels = []
    for i in range(n_images):
        filenames = path.basename(file_path[i])[0]
        filenames = int(filenames)
        if filenames == 0:
            filename = [1., 0.]
        else:
            filename = [0., 1.]
        labels.append(filename)
    labels = np.asarray(labels)
    return (images, labels)


# train 데이터 만들기
def seperate_train_test(images, labels, rate):
    train_test_split = rate
    n_images = images.shape[0]
    split_index = int(train_test_split * n_images)
    shuffled_indices = np.random.permutation(n_images)
    train_indices = shuffled_indices[0:split_index]
    test_indices = shuffled_indices[split_index:]

    x_train = images[train_indices, :, :]
    y_train = labels[train_indices]
    x_test = images[test_indices, :, :]
    y_test = labels[test_indices]

    return (x_train, y_train, x_test, y_test)


fire_images, fire_labels = loard_img('fire_png_28')
fire_x_train, fire_y_train, fire_x_test, fire_y_test = seperate_train_test(fire_images, fire_labels, 0.7)

none_images, none_labels = loard_img('none_png_28')
none_x_train, none_y_train, none_x_test, none_y_test = seperate_train_test(none_images, none_labels, 0.7)

x_train = np.r_[fire_x_train, none_x_train]
y_train = np.r_[fire_y_train, none_y_train]
x_test = np.r_[fire_x_test, none_x_test]
y_test = np.r_[fire_y_test, none_y_test]

learning_rate = 0.001
keep_prob = tf.placeholder(tf.float32)

# x,y tensor 만들기
X = tf.placeholder(tf.float32, [None, 2352])
X_img = tf.reshape(X, [-1, 28, 28, 3])  # img 28x28x3 (color)
Y = tf.placeholder(tf.float32, [None, 2])  # fire/ Nonefire

#  X ImgIn shape=(?, 28, 28, 3)
with tf.name_scope("Layer1"):
    W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
    #    Conv     -> (?, 28, 28, 32)
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    #    Pool     -> (?, 14, 14, 32)
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    #  L1 ImgIn shape=(?, 14,14, 3)

with tf.name_scope("Layer2"):
    W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    #    Conv      ->(?, 14, 14, 64)
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    #    Pool      ->(?, 7, 7, 64)
    L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
    # L2 ImgIn shape=(?, 7, 7, 64)

with tf.name_scope("Layer3"):
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
    #    Conv      ->(?, 7, 7, 128)
    #    Pool      ->(?, 4, 4, 128)
    #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
        1, 2, 2, 1], padding='SAME')
    L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
    L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])
    # L3 ImgIn shape=(?, 128 * 4 * 4)

W4 = tf.get_variable("F.C", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
# L4 FC 4x4x128 inputs -> 625 outputs


W5 = tf.get_variable("Final.F.C", shape=[625, 2],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L4, W5) + b5
# W5 Final FC 625 inputs -> 10 outputs


with tf.name_scope("Cost"):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    tf.summary.scalar("Cost", cost)

with tf.name_scope("Optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

print('Learning started! Please Waite.')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# trnsorboard저장 및 실행
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("./Tensorboard2/firedetection2")
writer.add_graph(sess.graph)  # Show the graph

x_train_1 = x_train.reshape(len(x_train), 2352)
x_test_1 = x_test.reshape(len(x_test), 2352)
for step in range(1000):
    acc_val, summary, cost_val, _ = sess.run([accuracy, merged_summary, cost, optimizer],
                                             feed_dict={X: x_train_1, Y: y_train, keep_prob: 0.7})

    writer.add_summary(summary, global_step=step)

    if step % 100 == 0 or step == 1000:
        print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

print('Learning Finished!')

print('===============Classification===============\n')
print('Accuracy:', sess.run(accuracy, feed_dict={
    X: x_test_1, Y: y_test, keep_prob: 1}))

r = random.randint(0, len(x_test) - 1)
print("Label: ", sess.run(tf.argmax(y_test[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(logits, 1), feed_dict={X: x_test_1[r:r + 1], keep_prob: 1}))
print('\n============================================')