---
layout: post
title: "Improving Clustering with Disentanglement"
tags: representation-learning autoencoder clustering
date: 2020-07-19
---

> An accompanying article for the paper "Improving k-Means Clustering with Disentangled Internal Representations" by A.F. Agarap and A.P. Azcarraga presented at the 2020 International Joint Conference on Neural Networks (IJCNN)

## Background

In the context of our work, we define **disentanglement** as **how far
class-different data points from each other are**, **relative to class-similar data
points**. This is similar to the way the aforementioned term was treated in
[Frosst et al. (2019)](https://arxiv.org/abs/1902.01889) So, **maximizing disentanglement** during representation
learning means the **distance among class-similar data points** are **minimized**.

![](../../../images/background-disentanglement.png)

In doing so, it would **preserve** the **class memberships** of the **examples** from the
dataset, i.e. how **data points reside** in the **feature space** as a **function of
their classes or labels**. If the **class memberships** are **preserved**, we would have
a feature representation space in which a nearest neighbor classifier or a
clustering algorithm would perform well.

### Clustering
Clustering is a machine learning task that finds the **grouping** of **data points**
wherein the **points in a group share more similarities** among themselves relative
to points in a different group.

![](../../../images/clustering.png)

Like other machine learning algorithms, the
success of clustering algorithms relies on the choice of **feature
representation**. One representation may be superior than another with respect to
the dataset used. However, in deep learning, this is not the case since the
feature representations are **learned** as an **implicit task** of a **neural network**.

### Deep Clustering
And so, recent works such as [Deep Embedding Clustering or
DEC](https://arxiv.org/abs/1511.06335) and [Variational
Deep Embedding or VADE](https://arxiv.org/abs/1611.05148) in 2016, and [ClusterGAN](https://arxiv.org/abs/1809.03627) in 2018, took advantage of the
feature representation learning capability of neural networks.

<figure>
<picture>
<img src="../../../images/dec.png">
</picture>
<center>
<figcaption>Figure from DEC (Xie et al., 2016). The network structure of DEC. </figcaption>
</center>
</figure>

We will not discuss them in detail in this article, but the **fundamental
idea** among these works is **essentially** the **same**, and that is to **simultaneously learn**
the feature **representations** and the **cluster assignments** using a deep neural
network. This approach is known as **deep clustering**.

## Motivation
Although deep clustering methods learn the clustering assignment together with feature representations, what they do not explicitly set out to do is to preserve the class neighbourhood structure of the dataset. This serves as our motivation for our research, and that is can we preserve the class neighbourhood structure of the dataset and then perform clustering on the learned representation of a deep network.

In 2019, the Not Too Deep or N2D Clustering method was proposed wherein they learned a latent code representation of a dataset, in which they further searched for an underlying manifold using techniques such as t-SNE, Isomap, and UMAP. The resulting manifold is a clustering-friendly representation of the dataset. So, after manifold learning, they used the learned manifold as the dataset features for clustering. Using this approach, they were able to have a good clustering performance. The N2D is a relatively simpler approach compared to deep clustering algorithms, and we propose a similar approach.

## Learning Disentangled Representations
We also use an autoencoder network to learn the latent code representation of a dataset, and then use the representation for clustering. We draw the line of difference on how we learn a more clustering-friendly representation. Instead of using manifold learning techniques, we propose to disentangle the learned representations of an autoencoder network.

To disentangle the learned representations, we use the soft nearest neighbour loss or SNNL which measures the entanglement of class-similar data points. What this loss function does is it minimizes the distances among class-similar data points in each of the hidden layer of a neural network. The work by Frosst, Papernot, and Hinton on this loss function used a fixed temperature value denoted by T. The temperature dictates how to control the importance given to the distances between pairs of points, for instance, at low temperatures, the loss is dominated by small distances while actual distances between widely separated representations become less relevant. They used SNNL for discriminative and generative tasks in their 2019 paper.

In our work, we used SNNL for clustering, and we introduce the use of an annealing temperature instead of a fixed temperature. Our annealing temperature is an inverse function with respect to the training epoch number which is denoted by tau.

Running a gradient descent on a randomly sampled and labelled 300 data points from a Gaussian distribution, we can see that using our annealing temperature for SNNL, we found faster disentanglement compared to using a fixed temperature. As we can see, even as early as the 20th epoch, the class-similar data points are more clustered together or entangled when using an annealing temperature than when using a fixed temperature, as it is also numerically shown by the SNNL value.

## Our Method
So, our contributions are the use of SNNL for disentanglement of feature representations for clustering, the use of an annealing temperature for SNNL, and a simpler clustering approach compared to deep clustering methods.
Our method can be summarized in the following manner,
1. We train an autoencoder with a composite loss of binary cross entropy as the reconstruction loss, and the soft nearest neighbour loss as a regularizer. The SNNL for each hidden layer of the autoencoder is minimized to preserve the class neighbourhood structure of the dataset.
2. After training, we use the latent code representation of a dataset as the dataset features for clustering.

## Clustering on Disentangled Representations
Our experiment configuration is as follows,
1. We used the MNIST, Fashion-MNIST, and EMNIST Balanced benchmark datasets. Each image in the datasets were flattened to a 784-dimensional vector. We used their ground-truth labels as the pseudo-clustering labels for measuring the clustering accuracy of our model.
2. We did not perform hyperparameter tuning or other training tricks due to computational constraints and to keep our approach simple.
3. Other regularizers like dropout and batch norm were omitted since they might affect the disentangling process.
4. We computed the average performance of our model across four runs, each run having a different random seed.

### Clustering Performance
However, autoencoding and clustering are both unsupervised learning tasks, while we use SNNL, a loss function that uses labels to preserve the class neighbourhood structure of the dataset.
With this in mind, we simulated the lack of labelled data by using a small subset of the labelled training data of the benchmark datasets. The number of labelled examples we used were arbitrarily chosen.
We retrieved the reported clustering accuracy of DEC, VaDE, ClusterGAN, and N2D from literature as baseline results, and in this slide, we can see the summary of our findings where our approach outperformed the baseline models.
Note that these results are the best clustering accuracy among the four runs for each dataset since the baseline results from literature are also the reported best clustering accuracy by the respective authors.

### Visualizing Disentangled Representations
To further support our findings, we visualized the disentangled representations by our network for each of the dataset.
For the EMNIST Balanced dataset, we randomly chose 10 classes to visualize for easier and cleaner visualization.
From these visualizations, we can see that the latent code representation for each dataset indeed became more clustering-friendly by having well-defined clusters as indicated by the cluster dispersion.

### Training on Fewer Labelled Examples
We also tried training our model on fewer labelled examples.
In this slide, we can see that even with fewer labelled training examples, the clustering performance on the disentangled representations is still on par with our baseline models from the literature.

## Conclusion
Compared to deep clustering methods, we employed a simpler clustering approach by using a composite loss of autoencoder reconstruction loss and SNNL to learn a more clustering-friendly representation that improves the performance of a k-Means clustering algorithm.
Our expansion on SNNL used an annealing temperature which helps with faster and better disentanglement that helped improve the clustering performance on the benchmark datasets. Thus concluding our work.
