---
title: "Improving k-Means Clustering Performance with Disentangled Internal Representations"
collection: publications
permalink: /publication/2020-07-19-clustering-ae
excerpt: "TLDR: Preserved class neighbourhood structure of the dataset which improves clustering performance."
date: 2020-07-20
venue: "International Joint Conference on Neural Networks (IJCNN)"
paperurl: "https://arxiv.org/abs/2006.04535"
---

**Abstract:** Deep clustering algorithms combine representation learning and clustering by jointly optimizing a clustering loss and a non-clustering loss. In such methods, a deep neural network is used for representation learning together with a clustering network. Instead of following this framework to improve clustering performance, we propose a simpler approach of optimizing the entanglement of the learned latent code representation of an autoencoder. We define entanglement as how close pairs of points from the same class or structure are, relative to pairs of points from different classes or structures. To measure the entanglement of data points, we use the soft nearest neighbor loss, and expand it by introducing an annealing temperature factor. Using our proposed approach, the test clustering accuracy was 96.2% on the MNIST dataset, 85.6% on the Fashion-MNIST dataset, and 79.2% on the EMNIST Balanced dataset, outperforming our baseline models. 


[Download paper here](https://arxiv.org/pdf/2006.04535.pdf)


Recommended citation: Agarap, A. F., & Azcarraga, A. P. (2020). Improving k-Means Clustering Performance with Disentangled Internal Representations. arXiv preprint arXiv:2006.04535.
