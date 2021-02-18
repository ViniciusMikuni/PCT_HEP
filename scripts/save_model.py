import argparse
import h5py
from math import *
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
import json
import os, ast
import sys

np.set_printoptions(threshold=sys.maxsize)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR,'..', 'models'))
sys.path.append(os.path.join(BASE_DIR,'..' ,'utils'))
import pct as MODEL



# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='../logs', help='Model checkpoint path')
parser.add_argument('--name', default='test', help='Output model name')
parser.add_argument('--num_point', type=int, default=100, help='Point Number [default: 400]')
parser.add_argument('--nfeat', type=int, default=16, help='Number of features [default: 5]')
parser.add_argument('--ncat', type=int, default=5, help='Number of categories [default: 2]')
parser.add_argument('--simple', action='store_true', default=False,help='Use simplified model')




FLAGS = parser.parse_args()
MODEL_PATH = os.path.join(FLAGS.model_path,FLAGS.name)




# MAIN SCRIPT
NUM_POINT = FLAGS.num_point
NFEATURES = FLAGS.nfeat
NUM_CATEGORIES = FLAGS.ncat

def load_pb(pb_model):
    with tf.gfile.GFile(pb_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph
  
def save_eval():
    with tf.Graph().as_default():
        pointclouds_pl, mask_pl,labels_pl = MODEL.placeholder_inputs(2, NUM_POINT,NFEATURES)
        print (pointclouds_pl.name,labels_pl.name)
                        
        is_training_pl = tf.placeholder(tf.bool, shape=())
        if FLAGS.simple:
            pred,att1,att2,att3 = MODEL.get_model_simple(pointclouds_pl,mask_pl, 
                                            is_training=is_training_pl,
                                            num_class=NUM_CATEGORIES,scname='PL')
        else:
            pred,att1,att2,att3 = MODEL.get_model(pointclouds_pl,mask_pl, is_training=is_training_pl,
                                                  num_class=NUM_CATEGORIES,scname='PL')
        pred = tf.nn.softmax(pred)
        saver = tf.train.Saver()
          
        
        config = tf.ConfigProto()
        sess = tf.Session(config=config)

        saver.restore(sess,os.path.join(MODEL_PATH,'model.ckpt'))
        print('model restored')

        builder = tf.saved_model.builder.SavedModelBuilder("../pretrained/{}".format(FLAGS.name))
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()


        #print([node.name for node in sess.graph.as_graph_def().node])
        print('prediction name:',pred.name)


            

################################################          
    
def load_eval():    
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], '../pretrained/{}'.format(FLAGS.name))
        flops = tf.profiler.profile(g, options=tf.profiler.ProfileOptionBuilder.float_operation())
        print('{} MFLOPS after freezing'.format(flops.total_float_ops/1e6))
        

if __name__=='__main__':
  save_eval()
  load_eval()
