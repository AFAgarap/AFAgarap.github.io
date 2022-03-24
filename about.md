---
layout: page
title: About
permalink: /about/
---

Hello! My name is Abien Fred Agarap, and I am an Artificial Intelligence
researcher focusing on deep learning and representation learning, with an
engineering background on computer vision, natural language processing, and
machine learning. I got my Master of Science in Computing Science degree
from De La Salle University - Manila (DLSU Manila), where I was supervised by
Dr. Arnulfo P. Azcarraga. I am currently working as a Team Lead at Augmented
Intelligence Pros (AI-Pros), Inc. Previously, I was the Lead Research Scientist
at Senti AI. Aside from my professional and academic careers, I contributed to
non-profit organizations that help strengthen pre-college scientific research
by having served as a Junior Academy Mentor at the New York Academy of
Sciences, and by having served in the Network of ISEF Alumni Philippines.

I was a recipient of the Intel Excellence in Computer Science award in 2013,
and was a part of the Philippine Team to the Intel International Science and
Engineering Fair (Intel ISEF) 2013 in Phoenix, Arizona, UA. In July 2018, I
qualified as one of the 24 participants out of 562 applicants worldwide (4.2%
acceptance rate) for the Google TensorFlow Deep Learning Camp in Jeju Island,
South Korea, where I was mentored by Dr. Yu-Han Liu, Developer Programs
Engineer for Machine Learning and Big Data at Google. I have published papers
in international conference proceedings by ACM, IEEE, and Springer.

Under Dr. Azcarraga's supervision, I worked on my thesis entitled "Self-Organizing
Cooperative Neural Network Experts", and it was given the Gold Medal
for the Most Outstanding Thesis Award at DLSU Manila.
     
If you are interested in communicating with me, my e-mail address is abienfred.agarap[at]gmail.com. Thank you for visiting!

## Education

- **Master of Science, De La Salle University - Manila**
<br>September 2018 to February 2022
<br>Thesis: Self-Organizing Cooperative Neural Network Experts
<br>Adviser: Dr. Arnulfo P. Azcarraga
<br>Award: Gold Medal, Most Outstanding Thesis Award
<br>
<br>Related published works:
<br>- Improving k-Means Clustering Performance with Disentangled Internal
Representations (IJCNN 2020, Acceptance rate: 26.43%)
<br>- k-Winners-Take-All Ensemble Neural Network (ICONIP 2021, Acceptance rate:
20.88%)
<br>- Mixture of Experts with Soft Nearest Neighbor Loss (Under review for
IJCNN 2022)

## Latest Publication

**k-Winners-Take-All Ensemble Neural Network**

Abien Fred Agarap, Arnulfo P. Azcarraga (ICONIP 2021, Acceptance rate: 20.88%)

**Abstract:** Ensembling is one approach that improves the performance of a neural network by combining a number of independent neural networks, usually by either averaging or summing up their individual outputs. We modify this ensembling approach by training the sub-networks concurrently instead of independently. This concurrent training of sub-networks leads them to cooperate with each other, and we refer to them as "cooperative ensemble". Meanwhile, the mixture-of-experts approach improves a neural network performance by dividing up a given dataset to its sub-networks. It then uses a gating network that assigns a specialization to each of its sub-networks called "experts". We improve on these aforementioned ways for combining a group of neural networks by using a k-Winners-Take-All (kWTA) activation function, that acts as the combination method for the outputs of each sub-network in the ensemble. We refer to this proposed model as "kWTA ensemble neural networks" (kWTA-ENN). With the kWTA activation function, the losing neurons of the sub-networks are inhibited while the winning neurons are retained. This results in sub-networks having some form of specialization but also sharing knowledge with one another. We compare our approach with the cooperative ensemble and mixture-of-experts, where we used a feed-forward neural network with one hidden layer having 100 neurons as the sub-network architecture. Our approach yields a better performance compared to the baseline models, reaching the following test accuracies on benchmark datasets: 98.34% on MNIST, 88.06% on Fashion-MNIST, 91.56% on KMNIST, and 95.97% on WDBC.<br>**Link:** [Springer](https://link.springer.com/chapter/10.1007%2F978-3-030-92270-2_22)


**Improving k-Means Clustering Performance with Disentangled Internal
Representations**

Abien Fred Agarap, Arnulfo P. Azcarraga (IJCNN 2020, Acceptance rate: 26.43%)

**Abstract:** Deep clustering algorithms combine representation learning and clustering by jointly optimizing a clustering loss and a non-clustering loss. In such methods, a deep neural network is used for representation learning together with a clustering network. Instead of following this framework to improve clustering performance, we propose a simpler approach of optimizing the entanglement of the learned latent code representation of an autoencoder. We define entanglement as how close pairs of points from the same class or structure are, relative to pairs of points from different classes or structures. To measure the entanglement of data points, we use the soft nearest neighbor loss, and expand it by introducing an annealing temperature factor. Using our proposed approach, the test clustering accuracy was 96.2% on the MNIST dataset, 85.6% on the Fashion-MNIST dataset, and 79.2% on the EMNIST Balanced dataset, outperforming our baseline models.<br>**Link:** [arXiv.org](https://arxiv.org/abs/2006.04535)


## Work Experience
- **Augmented Intelligence Pros, Inc.**<br>Since August 2019
- **Lead Research Scientist, Senti Techlabs Inc.**<br>August 2018 to July 2019
- **Mentee, Google Deep Learning Camp 2018**<br>Jeju, South Korea -- June 2018 to August 2018
- **Research Scientist, Senti Techlabs Inc.**<br>May 2018 to July 2018
- **AI Developer Apprentice, Senti Techlabs Inc.**<br>December 2017 to March 2018


## Projects

These projects were done as a pastime and as course projects.

**Text Classification and Clustering with Annealing Soft Nearest Neighbor Loss**<br>
Used the annealing soft nearest neighbor loss for AG News dataset classification and clustering.<br>**Link:** [findings](https://www.researchgate.net/publication/348050060_Text_Classification_and_Clustering_with_Annealing_Soft_Nearest_Neighbor_Loss), code (soon!).

**Statistical analysis on e-commerce reviews, with sentiment classification using bidirectional recurrent neural network (RNN)**<br>
Performed statistical analysis and sentiment classification on [women's e-commerce clothing reviews dataset](https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews).<br>**Links:** [arXiv.org](https://arxiv.org/abs/1805.03687), [code](https://github.com/AFAgarap/ecommerce-reviews-analysis).

**Towards building an intelligent anti-malware system: A deep learning approach using support vector machine (SVM) for malware classification**<br>
Developed deep learning models with SVM as the classifier, and used them for malware classification.<br>**Links:** [arXiv.org](https://arxiv.org/abs/1801.00318), [code](https://github.com/AFAgarap/malware-classification)

**An architecture combining convolutional neural network (CNN) and support vector machine (SVM) for image classification**<br>
Used a CNN as a feature extractor, and a SVM as the classifier for the MNIST and Fashion-MNIST dataset.<br>**Links:** [arXiv.org](https://arxiv.org/abs/1712.03541), [code](https://github.com/AFAgarap/cnn-svm)


**An application of machine learning algorithms on the Wisconsin diagnostic breast cancer (WDBC) dataset**<br>
An application of the gated recurrent neural network with SVM (GRNN-SVM) on the WDBC dataset.<br>**Links:** [arXiv.org](https://arxiv.org/abs/1711.07831), [code](https://github.com/AFAgarap/wisconsin-breast-cancer)
