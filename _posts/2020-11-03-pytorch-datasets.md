---
layout: post
title: "PyTorch Datasets"
description: ""
tags: pytorch datasets classification
date: 2020-11-03
---

I have been using the same sets of datasets due to the nature of my research
being more on the exploratory side rather than the real-world applications of deep
learning. In each of them, I have always written a somewhat _boilerplate_ code
for loading datasets and creating data loaders for my models.

So, I decided to write my own Python library for loading datasets and creating
data loaders for them, one which I can reuse in my projects. I use image
classification datasets for most of the models I try to learn, and/or to modify or
improve. Hence, I started filling in my library with the standard image
classification datasets such as [MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist), [Fashion-MNIST](https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist), [EMNIST-Balanced](https://pytorch.org/docs/stable/torchvision/datasets.html#emnist), [CIFAR10](https://pytorch.org/docs/stable/torchvision/datasets.html#cifar),
and [SVHN](https://pytorch.org/docs/stable/torchvision/datasets.html#svhn). But
in case I wanted to use other datasets, I started adding in non-image
classification datasets such as [AG News](http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html), [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/), and [Malware Classification](https://github.com/AFAgarap/malware-classification).

{% highlight python %}
from pt_datasets import load_dataset, create_dataloader

# load the training and test data
train_data, test_data = load_dataset(name="cifar10")

# create a data loader for the training data
train_loader = create_dataloader(
    dataset=train_data, batch_size=64, shuffle=True, num_workers=1
)

...

# use the data loader for training
model.fit(train_loader, epochs=10)
{% endhighlight %}

We can also encode the dataset features to a lower-dimensional space,

{% highlight python %}
import seaborn as sns
import matplotlib.pyplot as plt
from pt_datasets import load_dataset, encode_features

# load the training and test data
train_data, test_data = load_dataset(name="fashion_mnist")

# get the numpy array of the features
# the encoders can only accept np.ndarray types
train_features = train_data.data.numpy()

# flatten the tensors
train_features = train_features.reshape(
    train_features.shape[0], -1
)

# get the labels
train_labels = train_data.targets.numpy()

# get the class names
classes = train_data.classes

# encode training features using t-SNE with CUDA
encoded_train_features = encode_features(
    features=train_features,
    seed=1024,
    use_cuda=True,
    encoder="tsne"
)

# use seaborn styling
sns.set_style("darkgrid")

# scatter plot each feature w.r.t class
for index in range(len(classes)):
    plt.scatter(
        encoded_train_features[train_labels == index, 0],
        encoded_train_features[train_labels == index, 1],
        label=classes[index],
        edgecolors="black"
    )
plt.legend(loc="upper center", title="Fashion-MNIST classes", ncol=5)
plt.show()
{% endhighlight %}
