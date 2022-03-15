import argparse
from datetime import datetime
import numpy as np
import tensorflow as tf
import os,ast
import sys
import time
from sklearn import metrics
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,'..', 'models')) 
sys.path.append(os.path.join(BASE_DIR,'..', 'utils')) 
import pct_best_tf2 as MODEL
import provider_tf2 as provider

parser = argparse.ArgumentParser()


parser.add_argument('--model', default='pct_best_tf2', help='Model name [default: pct]')
parser.add_argument('--log_dir', default='bes_dev', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=100, help='Point Number  [default: 100]')
parser.add_argument('--num_bes', type=int, default=142, help='Number of BES variables  [default: 142]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate [default: 0.001]')

parser.add_argument('--data_dir', default='../h5', help='directory with data [default: hdf5_data]')
parser.add_argument('--nfeat', type=int, default=4, help='Number of features PF [default: 16]')
parser.add_argument('--ncat', type=int, default=2, help='Number of categories [default: 2]')
parser.add_argument('--sample', default='bes', help='sample to use')
parser.add_argument('--simple', action='store_true', default=False,help='Use simplified model')
parser.add_argument('--test', action='store_true', default=False,help='start a test training')



FLAGS = parser.parse_args()
DATA_DIR = FLAGS.data_dir
SAMPLE = FLAGS.sample
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FEAT = FLAGS.nfeat
NUM_BES = FLAGS.num_bes
NUM_CLASSES = FLAGS.ncat
MAX_EPOCH = FLAGS.max_epoch
LEARNING_RATE = FLAGS.learning_rate

MODEL_FILE = os.path.join(BASE_DIR, '..', 'models',FLAGS.model+'.py')
LOG_DIR = os.path.join('../logs',FLAGS.log_dir)

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_pct_best.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


LEARNING_RATE_CLIP = 1e-6
EARLY_TOLERANCE=15

TRAIN_FILE = os.path.join(DATA_DIR, 'HHSample_2017_BESTinputs_validation_flattened_standardized_tiny.h5')
train_data = provider.load_bes(TRAIN_FILE,batch_size=BATCH_SIZE)
TEST_FILE = os.path.join(DATA_DIR, 'HHSample_2017_BESTinputs_validation_flattened_standardized_tiny.h5')
test_data = provider.load_bes(TEST_FILE,batch_size=BATCH_SIZE)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def BEST_PCT():
    input_global = Input((NUM_BES,))
    if FLAGS.simple:
        pointclouds_lab,pred_lab =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_higgs,pred_higgs =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_bottom,pred_bottom =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_top,pred_top =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_W,pred_W =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_Z,pred_Z =MODEL.PCT_simple(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
    else:
        pointclouds_lab,pred_lab =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_higgs,pred_higgs =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_bottom,pred_bottom =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_top,pred_top =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_W,pred_W =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)
        pointclouds_Z,pred_Z =MODEL.PCT(input_global,NUM_POINT,NUM_FEAT,NUM_CLASSES)

    net = tf.concat([pred_lab,pred_higgs,pred_bottom,pred_top,pred_W,pred_Z],-1)
    net = Dense(256 ,activation='relu')(net)    
    net = Dropout(0.2)(net)
    net = Dense(128 ,activation='relu')(net)    
    net = Dropout(0.2)(net)
    outputs = Dense(NUM_CLASSES,activation='softmax')(net)

    return input_global,pointclouds_lab,pointclouds_higgs,pointclouds_bottom,pointclouds_top,pointclouds_W,pointclouds_Z,outputs

inputs = BEST_PCT()
outputs = inputs[-1]
inputs = inputs[:-1]


#Callbacks
callbacks=[
    EarlyStopping(patience=10,restore_best_weights=True),
    ReduceLROnPlateau(patience=5, verbose=1),
    ModelCheckpoint(os.path.join(LOG_DIR, 'model.ckpt'),
                    save_best_only=True,mode='auto',period=1,save_weights_only=True)
]

        
model = Model(inputs=inputs,outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),              
              metrics=['accuracy'],
)

hist =  model.fit(train_data,
                  epochs=100,
                  validation_data=test_data,
                  callbacks=callbacks,
)
