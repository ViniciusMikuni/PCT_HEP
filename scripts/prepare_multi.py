import h5py
import os, sys
import numpy as np
from optparse import OptionParser



def save_voxels(sample,NPARTS=100):
    ''' Prepare the samples to be used during training. 
    sample: raw h5 file
    NPARTS: Number of particles to save per event
    '''

    data = sample['jetConstituentList'][:]
    labels = sample['jets'][:,53:]


    keep_mask = ((labels[:,0]==1)|(labels[:,1]==1)|(labels[:,2]==1)|(labels[:,3]==1)|(labels[:,4]==1))
    data=data[keep_mask]
    labels=labels[keep_mask]


    features=np.concatenate(
        (
            np.expand_dims(data[:,:,8],axis=-1), 
            np.expand_dims(data[:,:,11],axis=-1),             
            np.expand_dims(np.ma.log(data[:,:,5]).filled(0),axis=-1), 
            np.expand_dims(np.ma.log(data[:,:,0]).filled(0),axis=-1), 
            np.expand_dims(np.ma.log(data[:,:,1]).filled(0),axis=-1), 
            np.expand_dims(np.ma.log(data[:,:,2]).filled(0),axis=-1), 
            np.expand_dims(np.ma.log(data[:,:,3]).filled(0),axis=-1),             
            np.expand_dims(data[:,:,4],axis=-1),
            np.expand_dims(data[:,:,6],axis=-1),
            np.expand_dims(data[:,:,7],axis=-1),
            np.expand_dims(data[:,:,9],axis=-1),
            np.expand_dims(data[:,:,10],axis=-1),
            np.expand_dims(data[:,:,12],axis=-1),
            np.expand_dims(data[:,:,13],axis=-1),
            np.expand_dims(data[:,:,14],axis=-1),
            np.expand_dims(data[:,:,15],axis=-1),
            
     ),axis=-1)

    features[np.abs(features)==np.inf] = 0
    labels = np.concatenate(
        (
            np.expand_dims(labels[:,0],axis=-1), #g
            np.expand_dims(labels[:,1],axis=-1), #q
            np.expand_dims(labels[:,2],axis=-1), #Z
            np.expand_dims(labels[:,3],axis=-1), #W
            np.expand_dims(labels[:,4],axis=-1), #top
        ),axis=-1)

    ivoxel = 0

    
    return features,labels
    
    
   


if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--nparts", type=int, default=100, help="Number of particles per event")
    parser.add_option("--make_eval", action="store_true", default=False, help="Only produce evaluation sample. Otherwise will produce train/test samples")
    parser.add_option("--dir", type="string", default="../samples/", help="Folder containing the input files")
    parser.add_option("--out", type="string", default="../h5/", help="Folder to save output files")

    (flags, args) = parser.parse_args()


    NPARTS=flags.nparts
    make_eval = flags.make_eval
    samples_path = flags.dir
    save_path = flags.out


    
    #Assuming that the 2 samples were saved under these respective folders
    if make_eval:
        samples_path = os.path.join(samples_path,'val')
    else:
        samples_path = os.path.join(samples_path,'train')

    files = os.listdir(samples_path)
    files = [f for f in files if f.endswith(".h5")]
    files = [f for f in files if '100p' in f] #I'm using the 100p samples, but could be changed if a different one was used

    pids = np.array([])
    ncount = 0

    for f in files:
        data = h5py.File(os.path.join(samples_path,f),"r")
        if 'jetConstituentList' in data.keys():
            feat,pid= save_voxels(data,NPARTS)
            if pids.size==0:
                pids = pid
                feats = feat
            else:
                pids = np.concatenate((pids,pid),axis=0)
                feats = np.concatenate((feats,feat),axis=0)



    NTRAIN = int(0.8*len(pids)) #80% of the data is used for training
    if make_eval:
        with h5py.File(os.path.join(save_path,"eval_top_{}P_Jedi.h5".format(NPARTS)), "w") as fh5: 
            dset = fh5.create_dataset("data", data=feats)
            dset = fh5.create_dataset("pid", data=pids)

    else:

        with h5py.File(os.path.join(save_path,"train_top_{}P_Jedi.h5".format(NPARTS)), "w") as fh5:#         
            dset = fh5.create_dataset("data", data=feats[:NTRAIN])
            dset = fh5.create_dataset("pid", data=pids[:NTRAIN])
            
        with h5py.File(os.path.join(save_path,"test_top_{}P_Jedi.h5".format(NPARTS)), "w") as fh5: #        
            dset = fh5.create_dataset("data", data=feats[NTRAIN:])
            dset = fh5.create_dataset("pid", data=pids[NTRAIN:])


            

