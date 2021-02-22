
import pandas as pd
import h5py
import os
import numpy as np
from optparse import OptionParser
    

def convert_coordinate(data):
    px = data[:,:,1]
    py = data[:,:,2]
    pz = data[:,:,3]
    energy  = data[:,:,0]

    
    phi = np.ma.arctan(np.divide(py,px, out=np.zeros_like(py), where=px!=0)).filled(0)
    pt = px/np.cos(phi)
    eta = np.ma.arcsinh(np.divide(pz,pt, out=np.zeros_like(pz), where=pt!=0)).filled(0)


    return np.abs(pt),eta,phi,energy

def convert_back(px,py,pz):
    phi = np.arctan(py/px)
    pt = px/np.cos(phi)
    eta = np.arcsinh(pz/pt)

        
    return np.abs(pt),eta,phi



def clustering_sum(data,output_name,nevents=1000,nparts=100):

    npid = data[:nevents,-1]
    

    particles = data[...,0:200*4]
    particles=particles.reshape((data.shape[0],-1,4))
    particles=particles[:nevents,:100,:]

    jets = np.sum(particles,axis=1)
    jets_energy = jets[:,0]
    jets_pt,jets_eta,jets_phi = convert_back(jets[:,1],jets[:,2],jets[:,3])
    NFEAT=7
    points = np.zeros((particles.shape[0],particles.shape[1],NFEAT))
    pt,eta,phi,energy = convert_coordinate(particles)

    points[:,:,0] = (eta - np.expand_dims(jets_eta,-1))*(eta!=0)
    points[:,:,1] = (phi - np.expand_dims(jets_phi,-1))*(eta!=0)
    points[:,:,2] = np.ma.log(pt)
    points[:,:,3] = np.ma.log(energy)
    points[:,:,4] = np.ma.log(energy/np.expand_dims(jets_energy,-1)) 
    points[:,:,5] = np.ma.log(pt/np.expand_dims(jets_pt,-1))
    points[:,:,6] = np.sqrt((eta - np.expand_dims(jets_eta,-1))**2 + (phi - np.expand_dims(jets_phi,-1))**2)*(eta!=0)
    

    with h5py.File(output_name, "w") as fh5:
        dset = fh5.create_dataset("data", data=points)
        dset = fh5.create_dataset("pid", data=npid)




if __name__=='__main__':
    parser = OptionParser(usage="%prog [opt]  inputFiles")
    parser.add_option("--npoints", type=int, default=100, help="Number of particles per event")
    parser.add_option("--folder", type="string", default='../data/TOPTAGGING', help="Folder containing input files")
    parser.add_option("--sample", type="string", default='val.h5', help="Input file name")
    parser.add_option("--dir", type="string", default="../h5/", help="Folder containing the input files")
    parser.add_option("--out", type="string", default="evaluate_ttbar.h5", help="Output file name")

    (flags, args) = parser.parse_args()
        

    samples_path = flags.folder
    sample = flags.sample
    output_name = flags.out 
    NPARTS = flags.npoints

    store = pd.HDFStore(os.path.join(samples_path,sample))
    data = store['table'].values
    
    clustering_sum(data,output_name,data.shape[0],NPARTS)




