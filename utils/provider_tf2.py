import tensorflow as tf
import h5py
import numpy as np

def load_bes(h5_filename,nevts=-1,batch_size=64):  
    f = h5py.File(h5_filename,'r')
    if nevts == -1:
        nevts =f['BES_vars'].shape[0]
    nevts=int(nevts)
    
    data=(f['BES_vars'][:nevts],
          f['LabFrame_PFcands'][:nevts],
          f['HiggsFrame_PFcands'][:nevts],
          f['BottomFrame_PFcands'][:nevts],
          f['TopFrame_PFcands'][:nevts],
          f['WFrame_PFcands'][:nevts],
          f['ZFrame_PFcands'][:nevts])

    #Will now convert to a tf dataset. should be more efficient than the previous strategy
    #Shuffling and batching is also automatic, so no need to shuffle again later

    dataset =tf.data.Dataset.from_tensor_slices(data)
    #label = f['pid'][:nevts].astype(int) #No stored in the tet file, will use a dummy instead
    label = np.random.randint(1, size=(nevts,2))
    dataset_label = tf.data.Dataset.from_tensor_slices(label)
    tf_data = tf.data.Dataset.zip((dataset, dataset_label)).shuffle(nevts).batch(batch_size)

    return tf_data

