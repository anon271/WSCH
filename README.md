# WSCH: Wide-Shallow Autoencoder for Self-Supervised Video Hashing with Cyclic Multi-Task Learning

![](figure/model.png)

![](figure/train.png)


## Catalogue <br> 
* [1. Getting Started](#getting-started)
* [2. Train](#train)
* [3. Test](#test)
* [4. Trained Models](#trained-models)
* [5. Results](#results)



## Getting Started

1\. Clone this repository:
```
git clone https://github.com/haungmozhi9527/ConMH.git
cd ConMH
```

2\. Create a conda environment and install the dependencies:
```
conda create -n conmh python=3.6
conda activate conmh
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch -c conda-forge
pip install -r requirements.txt
```

3\. Download Datasets: VGG features of FCVID and YFCC are kindly uploaded by the authors of [SSVH]. ResNet50 features of ActivityNet are kindly provided by the authors of [BTH]. You can download them from Baiduyun disk. 

| *Dataset* | *Link* |
| ---- | ---- |
| FCVID | [Baidu disk](https://pan.baidu.com/s/1v0qo4PtiZgFB9iLmj3sJIg?pwd=0000) |
| ActivityNet | [Baidu disk](https://pan.baidu.com/s/1cDJ0-6T2-AOeLgp5rBihfA?pwd=0000) |
| YFCC | [Baidu disk](https://pan.baidu.com/s/1jpqcRRFdiemGvlPpukxJ6Q?pwd=0000) |

4\. 在对应的Json文件中 (./Json/Anet.py ).

## Train

To train WSCH:


## Test

To test WSCH:


## Trained Models

We provide trained WSMH checkpoints. You can download them from Baiduyun disk: [Baidu disk](https://pan.baidu.com/s/1qdCe6eZQR6ijhen_MbDbUg?pwd=mfok#list/path=%2F) 

## Results

For this repository, the expected performance is:

| *Dataset* | *Bits* | *mAP@5* | *mAP@20* | *mAP@40* | *mAP@60* | *mAP@80* | *mAP@100* |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| FCVID | 16 | 0.350 | 0.252 | 0.216 | 0.196 | 0.181 | 0.169 |
| FCVID | 32 | 0.476 | 0.332 | 0.287 | 0.263 | 0.245 | 0.230 |
| FCVID | 64 | 0.524 | 0.373 | 0.326 | 0.301 | 0.283 | 0.267 |
| ActivityNet | 16 | 0.156 | 0.081 | 0.050 | 0.036 | 0.029 | 0.024 |
| ActivityNet | 32 | 0.229 | 0.124 | 0.075 | 0.054 | 0.042 | 0.035 |
| ActivityNet | 64 | 0.267 | 0.150 | 0.092 | 0.066 | 0.051 | 0.042 |
| YFCC | 16 | 0.225 | 0.146 | 0.122 | 0.113 | 0.108 | 0.104 |
| YFCC | 32 | 0.341 | 0.182 | 0.148 | 0.135 | 0.128 | 0.123 |
| YFCC | 64 | 0.368 | 0.194 | 0.158 | 0.143 | 0.135 | 0.130 |



[SSVH]:https://github.com/lixiangpengcs/Self-Supervised-Video-Hashing

[BTH]:https://github.com/Lily1994/BTH


