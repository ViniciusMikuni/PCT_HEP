import tensorflow as tf
import math
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util


def placeholder_inputs(batch_size, num_point, num_features):
    pointclouds_pf = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_features))
    mask_padded = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pf,mask_padded,  labels_pl

def GetLocalFeat(pc,scname,outsize,is_training,bn_decay):
    '''Return local features from embedded point cloud
    Input: point cloud shaped as (B,N,k,NFEAT)
    '''
    features = tf_util.conv2d(pc, outsize, [1,1], padding='VALID', 
                              stride=[1,1],activation_fn=tf.nn.relu,
                              bn=True, is_training=is_training, 
                              scope=scname+'local1',bn_decay=bn_decay)
    features = tf_util.conv2d(features, outsize, [1,1], padding='VALID',
                            stride=[1,1],activation_fn=tf.nn.relu,
                            bn=True, is_training=is_training,
                            scope=scname+'local2',bn_decay=bn_decay)
    
    features = tf.reduce_mean(features, axis=-2, keep_dims=True)
    
    return features



def GetSelfAtt(pc,mask,scname,outsize,is_training,bn_decay):
    '''Get the self-attention layer
    Input: 
          Point cloud with shape (batch_size,num_point,num_dims)
          Zero-padded Mask with shape (batch_size,num_point)
    Return:
          Offset attention with shape (batch_size,num_point,outsize)
    '''
                                        
    
    query = tf_util.conv1d_nobias(pc,outsize//4,1, padding='VALID',
                                  stride=1,activation_fn=None,
                                  bn=True, is_training=is_training,
                                  scope=scname+'query')
    

    key = tf_util.conv1d_nobias(pc,outsize//4, 1, padding='VALID',
                                stride=1,activation_fn=None,bn=True,
                                is_training=is_training, scope=scname+'key')

    key = tf.transpose(key,perm=[0,2,1]) #B,C//4,N

    value = tf_util.conv1d_nobias(pc, outsize, 1, padding='VALID',
                           stride=1,activation_fn=None,bn=True,
                           is_training=is_training, scope=scname+'value',
                           bn_decay=bn_decay)
    
    
    value = tf.transpose(value,perm=[0,2,1]) #B,C,N

    energy = tf.matmul(query,tf.squeeze(key)) #B,N,N    

    #Make zero-padded less important

    mask_offset = -1000*mask+tf.ones_like(mask)
    mask_matrix = tf.matmul(tf.expand_dims(mask_offset,-1),tf.transpose(tf.expand_dims(mask_offset,-1),perm=[0,2,1]))
    mask_matrix = mask_matrix - tf.ones_like(mask_matrix)
    energy = energy + mask_matrix
    attention = tf.nn.softmax(energy)
    zero_mask = tf.where(tf.equal(mask_matrix,0),tf.ones_like(mask_matrix),tf.zeros_like(mask_matrix))  

    attention = attention*zero_mask

    attention = attention / (1e-9 + tf.reduce_sum(attention,1, keepdims=True))
    self_att = tf.matmul(value,attention) #B,C,N
    self_att = tf.transpose(self_att,perm=[0,2,1]) #B,N,C
    
    self_att = tf_util.conv1d_nobias(pc-self_att,outsize, 1, 
                              padding='VALID', stride=1,
                              activation_fn=tf.nn.relu,bn=True, 
                              is_training=is_training, scope=scname+'att',
                              bn_decay=bn_decay)
    
    return pc+self_att,attention


def get_model(point_cloud, mask,is_training, num_class,
              weight_decay=None, bn_decay=None,scname=''):
    batch_size = point_cloud.get_shape()[0]
            
    k = 20
    adj,mask_matrix = tf_util.pairwise_distanceR(point_cloud[:,:,:3],mask)    
    nn_idx = tf_util.knn(adj, k=k)    

    edge_feature_0 = get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)    
    features_0 = GetLocalFeat(edge_feature_0,scname+'local0',128,is_training,bn_decay=bn_decay)

    adj = tf_util.pairwise_distance(features_0,mask_matrix)
    nn_idx = tf_util.knn(adj, k=k)

    edge_feature_1 = get_edge_feature(features_0, nn_idx=nn_idx, k=k)    
    features_1 = GetLocalFeat(edge_feature_1,scname+'local1',64,is_training,bn_decay=bn_decay)
    
    self_att_1,attention1 = GetSelfAtt(tf.squeeze(features_1),mask,scname+'att1',64,is_training,bn_decay=bn_decay)
    self_att_2,attention2 = GetSelfAtt(self_att_1,mask,scname+'att2',64,is_training,bn_decay=bn_decay)
    self_att_3,attention3 = GetSelfAtt(self_att_2,mask,scname+'att3',64,is_training,bn_decay=bn_decay)
    
    concat = tf.concat([
        self_att_1,
        self_att_2,
        self_att_3,    
        tf.squeeze(features_1),
     ]
        ,axis=-1)
    
    net = tf_util.conv1d(concat, 256, 1, padding='VALID',
                         stride=1,activation_fn=tf.nn.relu,
                         bn=True, is_training=is_training, 
                         scope=scname+'concat', bn_decay=bn_decay)

    net = tf.reduce_mean(net, axis=1, keep_dims=True)    
    net = tf.reshape(net, [batch_size, -1]) 

    net = tf_util.fully_connected(net, 128, bn=True, is_training=is_training,
                                  activation_fn=tf.nn.relu,
                                  scope=scname+'fc1',bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope=scname+'dp1')
    net = tf_util.fully_connected(net, num_class,activation_fn=None, scope='fc3'+scname)
    
    return net,attention1,attention2,attention3



def get_model_simple(point_cloud, mask,is_training, num_class,
                     weight_decay=None, bn_decay=None,scname=''):
    batch_size = point_cloud.get_shape()[0]    
    pc_transform = tf_util.conv1d(point_cloud, 128, 1, 
                                  padding='VALID', stride=1,
                                  activation_fn=tf.nn.relu,
                                  bn=True, is_training=is_training,
                                  scope=scname+'emb1', 
                                  bn_decay=bn_decay)


    pc_transform = tf_util.conv1d(pc_transform, 64, 1, 
                                  padding='VALID', stride=1,
                                  activation_fn=tf.nn.relu,
                                  bn=True, is_training=is_training,
                                  scope=scname+'emb2', 
                                  bn_decay=bn_decay)
    
    self_att_1,attention1 = GetSelfAtt(tf.squeeze(pc_transform),mask,scname+'att1',64,is_training,bn_decay=bn_decay)
    self_att_2,attention2 = GetSelfAtt(self_att_1,mask,scname+'att2',64,is_training,bn_decay=bn_decay)
    
    concat = tf.concat([
        self_att_1,
        self_att_2,    
    ]
       ,axis=-1)
   
    net = tf_util.conv1d(concat, 128, 1, padding='VALID',
                         stride=1,activation_fn=tf.nn.relu,
                         bn=False, is_training=is_training, 
                         scope=scname+'concat', bn_decay=bn_decay)

    net = tf.reduce_mean(net, axis=1, keep_dims=True)    
    net = tf.reshape(net, [batch_size, -1]) 

    net = tf_util.fully_connected(net, 64, bn=True, is_training=is_training,
                                 activation_fn=tf.nn.relu,
                                 scope=scname+'fc1',bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                       scope=scname+'dp1')
    
    net = tf_util.fully_connected(net, num_class,activation_fn=None, scope='fc3'+scname)
    
    return net,attention1,attention2,attention2


def get_edge_feature(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
    point_cloud: (batch_size, num_points, 1, num_dims) 
    nn_idx: (batch_size, num_points, k)
    k: int

    Returns:
    edge features: (batch_size, num_points, k, num_dims)
    """
    og_batch_size = point_cloud.get_shape().as_list()[0]
    point_cloud = tf.squeeze(point_cloud)
    if og_batch_size == 1:
        point_cloud = tf.expand_dims(point_cloud, 0)

    point_cloud_central = point_cloud

    batch_size,num_points,num_dims = point_cloud.get_shape()

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)


    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature







def get_loss(pred, label,num_class):
    """ pred: B,NUM_CLASSES
    label: B, """
    labels = tf.one_hot(indices=label, depth=num_class)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred)
    classify_loss = tf.reduce_mean(loss)        
    return classify_loss
