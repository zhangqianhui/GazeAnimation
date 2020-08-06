## GazeAnimation - Official Tensorflow Implementation

<p align="center"><img width="100%" src="img/animation.png" /></p>

> ***A Dual In-painting Model for Unsupervised Gaze Correction and Animation in the Wild***<br>
> Jichao Zhang, Jingjing Chen, [Hao Tang](https://ha0tang.github.io/), [Wei Wang](https://weiwangtrento.github.io/), [Yan Yan](https://userweb.cs.txstate.edu/~y_y34/), [Enver Sangineto](https://disi.unitn.it/~enver.sangineto/index.html), [Nicu Sebe](http://disi.unitn.it/~sebe/)<br>
> In ACMMM 2020.<br>

> Paper: https://arxiv.org/abs/1805.07509<br>

> **Abstract:** In this paper, we address the problem of unsupervised gaze correction in the wild that works without the need for precise annotations of the gaze angle and head pose. We have created a new NewGaze dataset which consists of two domains $X$, $Y$ in which eyes are either staring at the camera or somewhere else. Our method consists of three novel modules: Gaze Correction module~(GCM), Gaze Animation module~(GAM), and Pretrained Autoencoder module~(PAM). Specifically, GCM and GAM separately trains a dual in-painting model using data from domain $X$ for gaze correction and data from domain $Y$ for gaze animation. Additionally, a Synthesis-As-Training method is proposed when training GAM to encourage the features encoded from the eye region to be correlated with the angle information, resulting in gaze animation can be achieved by interpolation in the latent space. 
To further preserve the identity information~(\eg, eye shape, iris color), we propose the PAM with an Autoencoder by Self-Supervised mirror learning whose bottleneck features are angle-invariant and work as an extra input to the dual in-painting models. Extensive experiments validate the effectiveness of our model in gaze correction and gaze animation in the wild and demonstrate the superiority of our method in producing more compelling results than state-of-the-art baselines..*

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

## Usage

- Clone this repo:
```bash
git clone https://github.com/zhangqianhui/GazeAnimation.git
cd Sparsely-Grouped-GAN

```
- Download the CelebA dataset

You can download the [CelebA dataset](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAB06FXaQRUNtjW9ntaoPGvCa?dl=0) 
and unzip CelebA into a directory. 

 Just like
 
 ~~~
---------------------------------------------

The training data folder should look like : 
<base_dir/images/>
                |--image1
                |--image2
                |--image3
                |--image4...
    
list_attr_celeba.txt

---------------------------------------------
~~~

You can edit label_dir and data_dir in Baseoptions.txt and use the path for your CelebA dataset.

- Train the model using command line with python 
```bash
python train.py --exper_name='log20_5_1' --gpu_id='0' --loss_type='wgan_gp' --n_critic=5 --lam_c=10
```
- Test the model
```bash
python test.py --exper_name='log20_5_1' --gpu_id='0' 
```

Or Using scripts for training 
```bash
bash scripts/train_log20_5_1.sh
```
Using scripts for testing and pretained model can be downloaded [Pretrained Model](https://drive.google.com/file/d/1GYyGGQm06ahI0V4mBzSCaT1Fh4kzHb76/view?usp=sharing) 
```bash
bash scripts/test_log20_5_1.sh
```

## Experiment Result 

<p align="center"><img width="100%" src="img/results.jpeg" /></p>



