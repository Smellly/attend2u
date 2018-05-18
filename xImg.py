# -*- coding: utf-8 -*-
'''
// Author: Jay Smelly.
// Last modify: 2018-03-19 16:21:01.
// File name: img2text.py
//
// Description:
'''
import tensorflow as tf
slim = tf.contrib.slim
from PIL import Image
# from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import *
import numpy as np
import sys

def preprocess_image(image, width, height):
    im = tf.read_file(image)  
    im = tf.image.decode_jpeg(im)  
    im = tf.image.resize_images(im, (width, height))  
    im = tf.reshape(im, [-1, height, width, 3])  
    im = tf.cast(im, tf.float32) 
    return im

def extract_feature(image_path):
    tf.reset_default_graph()
    # checkpoint_file = 'model/resnet101/resnet_v2_101.ckpt'
    checkpoint_file = 'scripts/resnet_v1_101.ckpt'

    # image = tf.image.decode_jpeg(tf.read_file(image_path), channels=3)
    image_size = 224

    '''这个函数做了裁剪，缩放和归一化等'''
    processed_image = preprocess_image(image_path, 
                                        image_size, 
                                        image_size)

    '''Creates the Resnet V2 model.'''
    arg_scope = resnet_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = resnet_v1_101(processed_image, 
                                            num_classes=1000) #, \
                                            # is_training=False)   
        pool5 = tf.get_default_graph().get_operation_by_name('resnet_v1_101/pool5').outputs[0]
        pool5 = tf.transpose(pool5, perm=[0, 3, 1, 2])  # (batch_size, 2048, 1, 1)
        
    # probabilities = tf.nn.softmax(logits)

    saver = tf.train.Saver()
    with tf.Session() as sess:
	saver.restore(sess, checkpoint_file)
	_, logits, pool5_value = sess.run([processed_image, logits, pool5])
        # print(len(logits))
        # print(logits[0].shape)

    return pool5_value[0].astype(np.float32)

def add_context(np_path, feature):
    with open('data/caption_dataset/_test3.txt', 'r') as f:
        context = f.readlines()

    caps = []
    for line in context:
        caps.append(np_path.split('/')[-1] + line.split('.npy')[-1]) 
        # caps.append(line) 

    np_path = image_path.replace('.jpg', '.npy')
    # print('feature shape : ', feature.shape)
    # feature = tf.transpose(feature, perm=[0, 3, 1, 2])
    # feature = np.reshape(feature, (2048,1,1))
    print('feature shape : ', feature.shape)
    with open(np_path, 'w') as f:
        np.save(f, feature)

    with open('data/caption_dataset/test3.txt', 'w') as f:
        f.writelines(caps)

if __name__ == '__main__':
    image_path = sys.argv[1]
    feature = extract_feature(image_path)
    np_path = image_path.replace('.jpg', '.npy')
    add_context(np_path, feature)

