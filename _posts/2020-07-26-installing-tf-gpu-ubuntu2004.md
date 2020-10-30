---
layout: post
title: "Installing TensorFlow GPU in Ubuntu 20.04"
tags: tensorflow tutorial
description: "A short guide for installing TensorFlow GPU and its prerequisite packages in Ubuntu 20.04."
date: 2020-07-26
---

> A short guide for installing TensorFlow GPU and its prerequisite packages in Ubuntu 20.04.

When Ubuntu publishes a long-term support (LTS) release, I usually wait for a while before upgrading, mainly because I’m waiting for CUDA and cuDNN support for the new release. This time, it only took me three months to migrate from Ubuntu 18.04 to Ubuntu 20.04 — well, technically an Ubuntu-based distro, i.e. Regolith Linux. My decision to do so was simply because I upgraded my SSD from 120GB to 1TB, and so I migrated to a different OS as well— albeit just an Ubuntu derivative.

![](https://miro.medium.com/max/875/0*BZ9WK2GGWWHa0g32.jpg)

*Photo by [geordie_strike](https://pixabay.com/users/geordie_strike-17222189/) from [Pixabay](https://pixabay.com/photos/laptop-computer-business-tech-5421966/).*

As I expected, it took me a while to work things out in my new system. Fortunately, I saw some helpful answers online, and now I’m expanding on their answers by adding a bit more explanations. So, this post is actually based on the answers given by [meetnick](https://askubuntu.com/users/263979/meetnick) and [singrium](https://askubuntu.com/users/822295/singrium) in [this related question](https://askubuntu.com/questions/1230645/when-is-cuda-gonna-be-released-for-ubuntu-20-04) posted in [Ask Ubuntu](https://askubuntu.com/).

The installation of TensorFlow GPU in Ubuntu 20.04 can be summarized in the following points,
* Install CUDA 10.1 by installing nvidia-cuda-toolkit.
* Install the cuDNN version compatible with CUDA 10.1.
* Export CUDA environment variables.
* Install TensorFlow 2.0 with GPU support.

## Installing CUDA 10.1
First, ensure that you are using the NVIDIA proprietary driver by going to “Additional Drivers”, and then choosing the appropriate driver, i.e. for CUDA 10.1, the required driver version is ≥ 418.39. We use the proprietary version over the open source one since CUDA can only operate with the proprietary driver.

We are installing CUDA 10.1 because it is the compatible version with TensorFlow GPU.

![](https://miro.medium.com/max/875/0*M-eE2LPKX5trA4Hx.png)

*Image by the author. Choose the appropriate proprietary driver version for CUDA 10.1, i.e. ≥ 418.39.*

At the time of this writing, there is no available CUDA 10.1 for Ubuntu 20.04, but as meetnick points out in the referenced Ask Ubuntu post, installing `nvidia-cuda-toolkit` also installs CUDA 10.1.

![](https://miro.medium.com/max/875/1*mo5sI-Ek02Q9j42EnmhhWQ.png)

*Image by the author. The [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu).*

For the sake of being verbose, do not try to use 18.10 or 18.04 CUDA 10.1 for Ubuntu 20.04. I learned that the hard way, lol!

So, you can install CUDA 10.1 in Ubuntu 20.04 by running,

{% highlight shell %}
$ sudo apt install nvidia-cuda-toolkit
{% endhighlight %}

After installing CUDA 10.1, run `nvcc -V`. Then you will get an output similar to the following to verify if you had a successful installation,

{% highlight shell %}
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
{% endhighlight %}

Unlike in Ubuntu 18.04 (where I was from), CUDA is installed in a different path in 20.04, i.e. `/usr/lib/cuda` — which you can verify by running,

{% highlight shell %}
$ whereis cuda
cuda: /usr/lib/cuda /usr/include/cuda.h
{% endhighlight %}

In Ubuntu 18.04, as you might know, CUDA is installed in `/usr/local/cuda` or in `/usr/local/cuda-10.1`.

## Installing cuDNN

After installing CUDA 10.1, you can now install cuDNN 7.6.5 by downloading it from this [link](https://developer.nvidia.com/rdp/form/cudnn-download-survey). Then, choose “Download cuDNN”, and you’ll be asked to login or create an NVIDIA account. After logging in and accepting the terms of cuDNN software license agreement, you will see a list of available cuDNN software.

Click “Download cuDNN v7.6.5 (November 5th, 2019) for CUDA 10.1”, then choose “cuDNN Library for Linux” to download cuDNN 7.6.5 for CUDA 10.1. After downloading cuDNN, extract the files by running,

{% highlight shell %}
$ tar -xvzf cudnn-10.1-linux-x64-v7.6.5.32.tgz
{% endhighlight %}

Next, copy the extracted files to the CUDA installation folder,

{% highlight shell %}
$ sudo cp cuda/include/cudnn.h /usr/lib/cuda/include/
$ sudo cp cuda/lib64/libcudnn* /usr/lib/cuda/lib64/
{% endhighlight %}

Set the file permissions of cuDNN,

{% highlight shell %}
$ sudo chmod a+r /usr/lib/cuda/include/cudnn.h
/usr/lib/cuda/lib64/libcudnn*
{% endhighlight %}

## Export CUDA environment variables


The CUDA environment variables are needed by TensorFlow for GPU support. To set them, we need to append them to `~/.bashrc` file by running,

{% highlight shell %}
$ echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=/usr/lib/cuda/include:$LD_LIBRARY_PATH' >> ~/.bashrc
{% endhighlight %}

Load the exported environment variables by running,

{% highlight shell %}
$ source ~/.bashrc
{% endhighlight %}

## Installing TensorFlow 2.0

After installing the prerequisite packages, you can finally install TensorFlow 2.0,

{% highlight shell %}
$ pip install tensorflow
{% endhighlight %}

The `tensorflow` package now includes GPU support by default as opposed to the old days that we need to install `tensorflow-gpu` specifically.

Verify that TensorFlow can detect your GPU by running,

{% highlight python %}
>>> import tensorflow as tf
>>> tf.config.list_physical_devices("GPU")
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
{% endhighlight %}

If things went smoothly, you should have a similar output.

You can now enjoy using TensorFlow for your deep learning projects! Hooray!

---

If you are looking for a TensorFlow project to work on, perhaps you will find my blog on [Implementing Autoencoder in TensorFlow 2.0](https://afagarap.github.io/2019/03/20/implementing-autoencoder-in-tensorflow-2.0.html) enjoyable!

Also, if you enjoyed this article, perhaps you will enjoy my [other blogs](https://medium.com/@afagarap) as well!

This article was originally published at [Medium](https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d).
