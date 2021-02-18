# Point Cloud Transformers applied to Collider physics.

This is the main repository for the [PCT HEP paper](https://arxiv.org/abs/2001.05311).
The implementation uses a modified version of [PCT](https://arxiv.org/pdf/2012.09688.pdf) to suit the High Energy Physics needs.

# Requirements

[Tensorflow 1.14](https://www.tensorflow.org/)

[h5py](https://www.h5py.org/)

# Preparing the datasets
First, download the data for the application you want to test:

[Top quark dataset](https://zenodo.org/record/2603256)
[Quark/Gluon dataset](https://zenodo.org/record/3164691)
[Multiclassification dataset](https://zenodo.org/record/3602254)

To convert these files into the format required for the training, use the following scripts:

```bash
#Top quark dataset
python prepare_top.py --sample[val.h5/train.h5/test.h5] --out OUT_FILE_NAME

#Quark gluon dataset
python prepare_qg.py --out OUT_FILE_NAME

#Multiclassification dataset
python prepare_multi.py [--make_eval] --out OUT_FILE_NAME
```
For additional options, just run the scripts with the ```--help``` flag
To verify/change the name of the input files containing different datasets see the training script ```train_transformer.py```



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
A modified version of [PCT](https://arxiv.org/pdf/2012.09688.pdf) is used and implemented using the basic framework from [PointNet](https://github.com/charlesq34/pointnet).
