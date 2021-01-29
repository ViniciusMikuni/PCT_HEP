# Point Cloud Transformers applied to Collider physics.

This is the main repository for the [PCT HEP paper](https://arxiv.org/abs/2001.05311).
The implementation uses a modified version of [PCT](https://arxiv.org/pdf/2012.09688.pdf) to suit the High Energy Physics needs.
The input ```.h5``` files are expected to have the following structure:

* **data**: [N,P,F], 
* **pid**: [N]

N = Number of events
F = Number of features per point
P = Number of points

To verify/change the name of the input files containing different datasets see the training script ```train_transformer.py```

# Requirements

[Tensorflow 1.14](https://www.tensorflow.org/)

[h5py](https://www.h5py.org/)

# Training


```bash
cd classification
python train_transformer.py  --sample [qg/multi/top] [--simple] --log_dir OUTPUT_LOG
```
* --sample: training dataset to use
* --simple: activate this flag to change to simple PCT training 

A ```logs``` folder will be created with the training results under the main directory names OUTPUT_LOG.
To evaluate the training use:
```bash
python evaluate_transformer.py --sample [qg/multi/top] [--simple]  --model_path OUTPUT_LOG --batch 1000 --name OUTPUT_NAME 
```


# License

MIT License

# Acknowledgements
ABCNet uses a modified version of [PCT](https://arxiv.org/pdf/2012.09688.pdf) implemented using the basic framework from [PointNet](https://github.com/charlesq34/pointnet).