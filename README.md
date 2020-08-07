## GazeAnimation - Official Tensorflow Implementation

<p align="center"><img width="100%" src="img/animation.png" /></p>

> ***A Dual In-painting Model for Unsupervised Gaze Correction and Animation in the Wild***<br>
> Jichao Zhang, Jingjing Chen, [Hao Tang](https://ha0tang.github.io/), [Wei Wang](https://weiwangtrento.github.io/), [Yan Yan](https://userweb.cs.txstate.edu/~y_y34/), [Enver Sangineto](https://disi.unitn.it/~enver.sangineto/index.html), [Nicu Sebe](http://disi.unitn.it/~sebe/)<br>
> In ACMMM 2020.<br>

> Paper: https://arxiv.org/abs/1805.07509<br>

> **Abstract:** In this paper, we address the problem of unsupervised gaze correction in the wild that works without the need for precise annotations of the gaze angle and head pose. We have created a new NewGaze dataset which consists of two domains X, Y in which eyes are either staring at the camera or somewhere else. Our method consists of three novel modules: Gaze Correction module (GCM), Gaze Animation module (GAM), and Pretrained Autoencoder module (PAM). Specifically, GCM and GAM separately trains a dual in-painting model using data from domain X for gaze correction and data from domain Y for gaze animation. Additionally, a Synthesis-As-Training method is proposed when training GAM to encourage the features encoded from the eye region to be correlated with the angle information, resulting in gaze animation can be achieved by interpolation in the latent space. To further preserve the identity information (e.g., eye shape, iris color), we propose the PAM with an Autoencoder by Self-Supervised mirror learning whose bottleneck features are angle-invariant and work as an extra input to the dual in-painting models. Extensive experiments validate the effectiveness of our model in gaze correction and gaze animation in the wild and demonstrate the superiority of our method in producing more compelling results than state-of-the-art baselines.

### Citation

```
@inproceedings{zhang2018sparsely,
  title={Sparsely grouped multi-task generative adversarial networks for facial attribute manipulation},
  author={Zhang, Jichao and Shu, Yezhi and Xu, Songhua and Cao, Gongze and Zhong, Fan and Liu, Meng and Qin, Xueying},
  booktitle={Proceedings of the 26th ACM international conference on Multimedia},
  pages={392--401},
  year={2018}
}
```

## Network Architecture

![](img/model.png)

## Dependencies

```bash
Python=3.6
pip install -r requirments.txt

```
Or Using Conda

```bash
-conda create -name GazeA python=3.6
-conda install tensorflow-gpu=1.9
```
Other packages installed by pip.

## Usage

- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/GazeAnimation.git
cd GazeAnimation

```

- Download the NewGaze dataset

  Download the tar of NewGaze dataset from [Google Driver Linking](https://drive.google.com/open?id=1lYzpKdShN68RJGxRF1JgXnW-ved0F-mJ).
  
  ```bash
  cd your_path
  tar -xvf NewGazeData.tar
  ```
  
  Please edit the options.py and change your dataset path

- Pretraining Model

You can edit label_dir and data_dir

- Train the model using command line with python 
```bash
python train.py --use_sp --gpu_id='1' --exper_name='log8_7' --crop_w=50 --crop_h=30
```
- Test the model
```bash
python test.py --exper_name='log8_7' --gpu_id='0' --crop_h=30 --crop_w=50 --test_sample_dir='test_sample_dir' --checkpoints='checkpoints'
```

Or Using scripts for training 
```bash
bash scripts/train_log8_7.sh
```
Using scripts for testing and pretained model can be downloaded [Pretrained Model](https://drive.google.com/file/d/1GYyGGQm06ahI0V4mBzSCaT1Fh4kzHb76/view?usp=sharing) 
```bash
bash scripts/test_log8_7.sh
```

## Experiment Result 

### Gaze Correction

<p align="center"><img width="100%" src="img/correction.png" /></p>


### Gaze Animation

![](img/0.gif)
![](img/28.gif)
![](img/1.gif)
![](img/3.gif)
![](img/4.gif)
![](img/5.gif)
![](img/6.gif)
![](img/9.gif)
![](img/10.gif)
![](img/11.gif)
![](img/12.gif)
![](img/13.gif)
![](img/14.gif)
![](img/15.gif)
![](img/16.gif)
![](img/17.gif)
![](img/18.gif)
![](img/19.gif)
![](img/22.gif)
![](img/23.gif)
![](img/26.gif)
![](img/29.gif)


# Related Work

- [Sparsely_Grouped_GAN](https://github.com/zhangqianhui/Sparsely-Grouped-GAN)

- [GazeCorrection](https://github.com/zhangqianhui/GazeCorrection)

