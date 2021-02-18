import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser



def convert_back(px,py,pz):
    energy  = np.sqrt(px**2 + py**2 + pz**2)    
    phi = np.arctan(py/px)
    pt = px/np.cos(phi)
    eta = np.arcsinh(pz/pt)
        
    return pt,eta,phi,energy


def convert_coordinate(data):
    pt = data[:,:,0]
    eta = data[:,:,1]
    phi = data[:,:,2]
    
    px = np.abs(pt)*np.cos(phi)
    py = np.abs(pt)*np.sin(phi)
    pz = np.abs(pt)*np.sinh(eta)
    energy  = np.sqrt(px**2 + py**2 + pz**2)
        
    return px,py,pz,energy


def GetE(data):
    pt = data[:,:,0]
    eta = data[:,:,1]
    phi = data[:,:,2]

    px = np.abs(pt)*np.cos(phi)
    py = np.abs(pt)*np.sin(phi)
    pz = np.abs(pt)/(np.tan(2*np.arctan(np.exp(-eta))))
    energy  = np.sqrt(px**2 + py**2 + pz**2)
    
    return energy

def clustering_sum(data,nparts=100,out_dir):
    '''Just sum the 4 vector of the inputs'''
    npid = data['y']
    npdata = data['X']
    data_clip = []
    jets = []
    evts = []
    NFEAT=13
    print("Start clustering...")       
    for sample in npdata:
        px,py,pz,e = convert_coordinate(sample)            
        for ievt,evt in enumerate(sample):
            jets.append(convert_back(px[ievt].sum(),py[ievt].sum(),pz[ievt].sum()))
            
        data_clip.append(sample[:,:nparts])#Clip at nparts

    print("Adding clouds...")
    data_clip = np.concatenate(data_clip,axis=0)
    points = np.zeros((data_clip.shape[0],data_clip.shape[1],NFEAT))
    points[:,:,0] = data_clip[:,:,1] - np.expand_dims(np.array(jets)[:,1],-1) #eta
    points[:,:,1] = data_clip[:,:,2] - np.expand_dims(np.array(jets)[:,2],-1) #phi
    points[:,:,2] = np.ma.log(data_clip[:,:,0])  #log pt
    points[:,:,3] = np.ma.log(GetE(data_clip))  #log e
    points[:,:,4] = np.ma.log(GetE(data_clip)/np.expand_dims(np.array(jets)[:,3],-1))  #log e/jet e
    points[:,:,5] = np.ma.log(data_clip[:,:,0]/np.expand_dims(np.array(jets)[:,0],-1))  #log pt/jet pt
    points[:,:,6] = np.sqrt((data_clip[:,:,1] - np.expand_dims(np.array(jets)[:,1],-1))**2 + (data_clip[:,:,2] - np.expand_dims(np.array(jets)[:,2],-1))**2)
    points[:,:,7] = np.ma.divide(data_clip[:,:,3],np.abs(data_clip[:,:,3]))
    points[:,:,8] = np.abs(data_clip[:,:,3])==11
    points[:,:,9] = np.abs(data_clip[:,:,3])==13
    points[:,:,10] = np.abs(data_clip[:,:,3])==22
    points[:,:,11] = np.abs(data_clip[:,:,3])==130
    points[:,:,12] = np.abs(data_clip[:,:,3])==211


    samples = ['train','test','evaluate']
    event_start = 0
    event_end = split[0]
    for isamp, sample in enumerate(samples):
        data = points[event_start:event_end]
        label = npid[event_start:event_end] 
        with h5py.File('{}/{}_{}'.format(out_dir,sample,output_name), "w") as fh5:
            dset = fh5.create_dataset("data", data=data)
            dset = fh5.create_dataset("pid", data=label)
        if isamp < len(samples)-1:
            event_start += event_end
            event_end += split[isamp+1]




if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default="qg/PYTHIA", help="Folder containing input files")
    parser.add_option("--dir", type="string", default="../h5/", help="Folder to save output files")
    parser.add_option("--out", type="string", default="PYTHIA", help="Output file name")

    (flags, args) = parser.parse_args()

    samples_path = flags.folder


    files = []
    for r, d, f in os.walk(samples_path):
        for file in f:
            if '.npz' in file:
                files.append(os.path.join(r, file))
    print(files)

    output_name = flags.out




    NPARTS = flags.npoints
    split = [int(1.6e6),int(2e5),int(2e5)]




    
    print("Loading data...")
    data = {}
    for ifi, f in enumerate(files):
        dataset = np.load(f)
        for key in dataset.files:
            if ifi==0:
                if key == 'y':
                    data[key] = dataset[key]
                else:
                    data[key] = [dataset[key]]
            else:
                if key == 'y':
                    data[key] = np.concatenate((data[key],dataset[key]),axis=0)
                else:
                    data[key].append(dataset[key])



    clustering_sum(data,NPARTS,flags.dir)
