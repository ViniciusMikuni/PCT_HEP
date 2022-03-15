import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPool2D, Conv1D, BatchNormalization
from tensorflow.keras.models import Model
import numpy as np
import sys
    



def GetLocalFeat(pc,outsize):
    '''Return local features from embedded point cloud
    Input: point cloud shaped as (B,N,k,NFEAT)
    '''

    features = Conv2D(outsize, kernel_size=[1,1], data_format='channels_last',
                      strides=[1,1],activation='relu')(pc)
    #features = BatchNormalization()(features) 
    features = Conv2D(outsize, kernel_size=[1,1], data_format='channels_last',
                      strides=[1,1],activation='relu')(features)
    #features = BatchNormalization()(features) 
    features = tf.reduce_mean(features, axis=-2)    
    return features



def GetSelfAtt(pc,mask,outsize,nheads=1):
    '''Get the self-attention layer
    Input: 
          Point cloud with shape (batch_size,num_point,num_dims)
          Zero-padded Mask with shape (batch_size,num_point)
    Return:
          Offset attention with shape (batch_size,num_point,outsize)
    '''
    
    assert outsize%nheads==0, "Output size is not a multiple of the number of heads"
    
    batch_size= tf.shape(pc)[0]
    #nparts = tf.shape(pc)[1]
    nparts = pc.get_shape()[1]

    query = Conv1D(outsize, kernel_size = 1,use_bias=False,
                   strides=1,activation=None)(pc) #B,N,C
    if nheads>1:
        query = tf.reshape(query,[batch_size, nparts, outsize//nheads,nheads])
        query =tf.transpose(query,perm=[0,3,1,2])
    
    #query = BatchNormalization()(query) 
    key = Conv1D(outsize, kernel_size = 1,use_bias=False,
                 strides=1,activation=None)(pc)
    #key = BatchNormalization()(key)

    if nheads>1:
        key = tf.reshape(key,[batch_size, nparts, outsize//nheads,nheads])
        key =tf.transpose(key,perm=[0,3,2,1])
    else:
        key = tf.transpose(key,perm=[0,2,1]) #B,C,N

    value = Conv1D(outsize, kernel_size = 1,use_bias=False,
                   strides=1,activation=None)(pc)    
    #value = BatchNormalization()(value)

    if nheads>1:
        value = tf.reshape(value,[batch_size, nparts, outsize//nheads,nheads])
        value =tf.transpose(value,perm=[0,3,2,1])
    else:
        value = tf.transpose(value,perm=[0,2,1]) #B,C,N

    energy = tf.matmul(query,key) #B,N,N    

    #Make zero-padded less important

    mask_offset = -1000*mask+tf.ones_like(mask)
    mask_matrix = tf.matmul(tf.expand_dims(mask_offset,-1),tf.transpose(tf.expand_dims(mask_offset,-1),perm=[0,2,1]))
    mask_matrix = mask_matrix - tf.ones_like(mask_matrix)

    
    if nheads>1:
        mask_matrix = tf.expand_dims(mask_matrix,axis=1)
        mask_matrix = tf.tile(mask_matrix, [1, nheads, 1, 1])
    
    energy = energy + mask_matrix
    attention = tf.keras.activations.softmax(energy)
    zero_mask = tf.where(tf.equal(mask_matrix,0),tf.ones_like(mask_matrix),tf.zeros_like(mask_matrix))  

    if nheads>1:
        attention = attention / (1e-9 + tf.reduce_sum(attention,2, keepdims=True))
    else:
        attention = attention / (1e-9 + tf.reduce_sum(attention,1, keepdims=True))
        
    self_att = tf.matmul(value,attention) #B,C,N
    if nheads>1:
        self_att = tf.reshape(self_att,[batch_size, outsize,nparts])
        
    self_att = tf.transpose(self_att,perm=[0,2,1]) #B,N,C

    

    self_att = Conv1D(outsize, kernel_size = 1,use_bias=False,
                      strides=1,activation='relu')(pc-self_att)    
    #self_att = BatchNormalization()(self_att) 
    return pc+self_att,attention


def GetEdgeFeat(point_cloud, nn_idx, k=20):
    """Construct edge feature for each point
    Args:
    point_cloud: (batch_size, num_points, 1, num_dims) 
    nn_idx: (batch_size, num_points, k)
    k: int
    Returns:
    edge features: (batch_size, num_points, k, num_dims)
    """



    point_cloud_central = point_cloud

    batch_size = tf.shape(point_cloud)[0]
    num_points = tf.shape(point_cloud)[1]
    num_dims = point_cloud.get_shape()[2]



    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
    point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)


    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature

def pairwise_distanceR(point_cloud, mask=None):
    """Compute pairwise distance in the eta-phi plane for the point cloud.
    Uses the third dimension to find the zero-padded terms
    Args:
      point_cloud: tensor (batch_size, num_points, 2)
      IMPORTANT: The order should be (eta, phi)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    point_cloud = point_cloud[:,:,:2] #eta-phi
  
    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1]))
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose) # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True) # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])

    if mask != None:
        zero_mask = 10000*tf.expand_dims(mask,-1)
        zero_mask_transpose = tf.transpose(zero_mask, perm=[0, 2, 1])
        zero_mask = zero_mask + zero_mask_transpose
        zero_mask = tf.where(tf.equal(zero_mask,20000),tf.zeros_like(zero_mask),zero_mask)

    return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose + zero_mask,zero_mask 


def pairwise_distance(point_cloud, mask=None): 
    """Compute pairwise distance of a point cloud.
    Args:
      point_cloud: tensor (batch_size, num_points, num_dims)
    Returns:
      pairwise distance: (batch_size, num_points, num_points)
    """

    point_cloud_transpose = tf.transpose(point_cloud, perm=[0, 2, 1]) 
    point_cloud_inner = tf.matmul(point_cloud, point_cloud_transpose) # x.x + y.y + z.z shape: NxN
    point_cloud_inner = -2*point_cloud_inner
    point_cloud_square = tf.reduce_sum(tf.square(point_cloud), axis=-1, keepdims=True) # from x.x, y.y, z.z to x.x + y.y + z.z
    point_cloud_square_tranpose = tf.transpose(point_cloud_square, perm=[0, 2, 1])
    # is_zero = point_cloud[:,:,] 
    # point_shift = 1000*tf.where(tf.equal(pt,0),tf.ones_like(pt),tf.fill(tf.shape(pt), tf.constant(0.0, dtype=pt.dtype)))
    if mask != None:
        return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose + mask
    else:
        return point_cloud_square + point_cloud_inner + point_cloud_square_tranpose


def knn(adj_matrix, k=20):
    """Get KNN based on the pairwise distance.
    Args:
      pairwise distance: (batch_size, num_points, num_points)
      k: int

    Returns:
      nearest neighbors: (batch_size, num_points, k)
    """
    neg_adj = -adj_matrix
    _, nn_idx = tf.math.top_k(neg_adj, k=k)  # values, indices
    return nn_idx


def PCT(npoints,nvars=1,nclass=2,nheads=1,nglobal=1):
    inputs = Input((npoints,nvars))
    input_global = Input((nglobal,))
    batch_size = tf.shape(inputs)[0]
            
    k = 7
    
    mask = tf.where(inputs[:,:,2]==0,K.ones_like(inputs[:,:,2]),K.zeros_like(inputs[:,:,2]))
    adj,mask_matrix = pairwise_distanceR(inputs[:,:,:3],mask)

    nn_idx = knn(adj, k=k)
    edge_feature_0 = GetEdgeFeat(inputs, nn_idx=nn_idx, k=k)    
    features_0 = GetLocalFeat(edge_feature_0,32*nheads)

    adj = pairwise_distance(features_0,mask_matrix)
    nn_idx = knn(adj, k=k)

    edge_feature_1 = GetEdgeFeat(features_0, nn_idx=nn_idx, k=k)    
    features_1 = GetLocalFeat(edge_feature_1,16*nheads)
    
    self_att_1,attention1 = GetSelfAtt(features_0,mask,32*nheads,nheads)
    self_att_2,attention2 = GetSelfAtt(features_0 + self_att_1,mask,32*nheads,nheads)
    self_att_3,attention3 = GetSelfAtt(features_1 + self_att_2,mask,32*nheads,nheads)

    concat = tf.concat([
        self_att_1,
        self_att_2,
        self_att_3,    
        features_1,
     ]
        ,axis=-1)


    net = Conv1D(256, kernel_size = 1,strides=1,activation='relu')(concat)
    #net = BatchNormalization()(net) 
    net_mean = tf.reduce_mean(net, axis=1)
    net_max = tf.reduce_max(net, axis=1)

    #Process global inputs
    net_glob = Dense(256 ,activation='relu')(input_global)    
    net_glob = Dense(128 ,activation='relu')(net_glob)    
    net = tf.concat([net_mean,net_max,net_glob],axis=-1)
    #net = BatchNormalization()(net)

    net = Dense(128,activation='relu')(net)
    
    #net = BatchNormalization()(net) 
    net = Dropout(0.2)(net)
    net = Dense(64,activation='relu')(net)
    outputs = Dense(nclass,activation='softmax')(net)
    return inputs,input_global,outputs

def PCT_simple(npoints,inputs_global,nvars=1,nclass=2,nheads=1):
    inputs = Input((npoints,nvars))
    batch_size = tf.shape(inputs)[0]
    
    pc_transform = Conv1D(128, kernel_size = 1,strides=1,activation='relu')(inputs)
    pc_transform = Conv1D(64, kernel_size = 1,strides=1,activation='relu')(pc_transform)
    
    mask = tf.where(inputs[:,:,2]==0,K.ones_like(inputs[:,:,2]),K.zeros_like(inputs[:,:,2]))
    
    self_att_1,attention1 = GetSelfAtt(tf.squeeze(pc_transform),mask,32*nheads,nheads)
    self_att_2,attention2 = GetSelfAtt(tf.squeeze(pc_transform) + self_att_1,mask,32*nheads,nheads)

    concat = tf.concat([
        self_att_1,
        self_att_2,
     ]
        ,axis=-1)


    net = Conv1D(128, kernel_size = 1,strides=1,activation='relu')(concat)
    #net = BatchNormalization()(net) 
    net_mean = tf.reduce_mean(net, axis=1)
    net_max = tf.reduce_max(net, axis=1)

    #Process global inputs
    net_glob = Dense(256 ,activation='relu')(input_global)    
    net_glob = Dense(128 ,activation='relu')(net_glob)    
    net = tf.concat([net_mean,net_max,net_glob],axis=-1)
    #net = BatchNormalization()(net)

    net = Dense(128,activation='relu')(net)
    
    #net = BatchNormalization()(net) 
    net = Dropout(0.2)(net)
    net = Dense(64,activation='relu')(net)
    outputs = Dense(nclass,activation='softmax')(net)
    return inputs,outputs



if __name__ == '__main__':
    inputs,input_global,outputs = PCT(npoints=3,nvars=3,nheads=1,nglobal=1)
    model = Model(inputs=[inputs,input_global], outputs=outputs)
    
    test_points = np.array([
        [
            [1,2,3],[3,4,5],[5,6,7]
        ],
        
        [
            [1,2,8],[13,14,15],[0,0,0]
        ]
    ])
    global_vars = np.array([1,2])
    label = np.array([0,1])
    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  run_eagerly=True,
    )

    model.fit([test_points,global_vars],label,epochs=1)
    name_translation = {
        'tf.where':'initial mask',
        'tf.__operators__.add_3':'mask matrix',
        'tf.math.subtract_1': 'mask matrix attention',
        'tf.where_2': 'zero mask',
        'tf.math.multiply_4': 'attention',
        'tf.math.top_k':'knn idx'
    }
    


        
    for layer in model.layers:
        if layer.name in name_translation.keys():
            print(name_translation[layer.name])
            keras_function = K.function([model.input], [layer.output])
            print(keras_function([[test_points,global_vars], 1]))
    
   
    
