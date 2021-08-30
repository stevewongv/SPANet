# Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset (CVPR'19)
[Tianyu Wang](https://stevewongv.github.io)\*, Xin Yang\*, Ke Xu, Shaozhe Chen, Qiang Zhang, [Rynson W.H. Lau](http://www.cs.cityu.edu.hk/~rynson/) † 
(\* Joint first author. † Rynson Lau is the corresponding author.)

[\[Arxiv\]](https://arxiv.org/abs/1904.01538) 

## Abstract
Removing rain streaks from a single image has been drawing considerable attention as rain streaks can severely degrade the image quality and affect the performance of existing outdoor vision tasks. While recent CNN-based derainers have reported promising performances, deraining remains an open problem for two reasons. First, existing synthesized rain datasets have only limited realism, in terms of modeling real rain characteristics such as rain shape, direction and intensity. Second, there are no public benchmarks for quantitative comparisons on real rain images, which makes the current evaluation less objective. The core challenge is that real world rain/clean image pairs cannot be captured at the same time. In this paper, we address the single image rain removal problem in two ways. First, we propose a semi-automatic method that incorporates temporal priors and human supervision to generate a high-quality clean image from each input sequence of real rain images. Using this method, we construct a large-scale dataset of ∼29.5K rain/rain-free image pairs that cover a wide range of natural rain scenes. Second, to better cover the stochastic distributions of real rain streaks, we propose a novel SPatial Attentive Network (SPANet) to remove rain streaks in a local-to-global manner. Extensive experiments demonstrate that our network performs favorably against the state-of-the-art deraining methods.

## Citation
If you use this code or our dataset(including test set), please cite:

```
@InProceedings{Wang_2019_CVPR,
  author = {Wang, Tianyu and Yang, Xin and Xu, Ke and Chen, Shaozhe and Zhang, Qiang and Lau, Rynson W.H.},
  title = {Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
```

## Dataset
See [my personal site](https://stevewongv.github.io/)

**UPDATE** We release the [code](https://github.com/stevewongv/SPANet/blob/master/clean%20image%20generation.ipynb) of clean image generation. We also provide some [synthesize and real video examples](https://drive.google.com/file/d/1AgwSDy0W91uWGH9r6H4yMySFyJJh0vpM/view?usp=sharing) for researchers to try. Note that we only implemented the code using 8 threads.

## Requirements
* PyTorch == 0.4.1 (1.0.x may not work for training)
* cupy ([Installation Guide](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy))
* opencv-python
* TensorBoardX
* Python3.6
* progressbar2
* scikit-image
* ffmpeg >= 4.0.1
* python-ffmpeg

## Setup
* **Clone this repo:**

```git
$ git clone ...
$ cd SPANet
```

## Train & Test
**Train:**
* Download the dataset(~44GB) and unpack it into code folder (See details in `Train_Dataset_README.md`). Then, run:

```bash
$ python main.py -a train -m latest
```

**Test:**
* Download the test dataset(~455MB) and unpack it into code folder (See details in `Test_Dataset_README.md`). Then, run: 

```
$ python main.py -a test -m latest
```

## Performance Change

PSNR 38.02 -> 38.53

SSIM 0.9868 -> 0.9875

**For generalization, we here stop at 40K steps.**

**All PSNR and SSIM of results are computed by using `skimage.measure`. Please use this to evaluate your works.**



## License
Please see `License.txt` file.

## Acknowledgement 

Code borrows from [RESCAN](https://github.com/XiaLiPKU/RESCAN) by [Xia Li](https://github.com/XiaLiPKU). The CUDA extension references [pyinn](https://github.com/szagoruyko/pyinn) by [Sergey Zagoruyko](https://github.com/szagoruyko) and [DSC(CF-Caffe)](https://github.com/xw-hu/CF-Caffe) by [Xiaowei Hu](https://github.com/xw-hu). Thanks for sharing!

## Contact
E-Mail: steve.w.git@icloud.com
