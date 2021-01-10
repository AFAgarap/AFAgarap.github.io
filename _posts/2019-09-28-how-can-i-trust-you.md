---
layout: post
title: "How can I trust you? An intuition and tutorial on trust score"
image: ../../../images/ae_mnist.png
description: "We lightly probe the trustworthiness of the predictions given by
a machine learning model."
tags: tutorial machine-learning deep-learning neural-networks trust-score metrics
date: 2019-03-20
---

> We lightly probe the trustworthiness of the predictions given by
a machine learning model.

Several efforts to improve deep learning performance have been done through the
years, but there are only few works done towards better understanding the
models and their predictions, and whether they should be trusted or not.

In this article, we shall lightly probe the trustworthiness of a model in terms
of its predictions. However, the term trust might seem vague and might reflect
a wide range of its denotations and/or connotations. So, for the sake of our
discussion, it may be safer that we limit the term trust to denote a
"fail-safe" feature for a model's predictions — that is, a secondary or
supporting opinion of the model predictions.

> If you are more interested on the practical stuff, you may skip to the Trust Score section.


<figure>
<picture>
<img src="https://miro.medium.com/max/933/1*ivzFhAkG3PcjzgvcvE5IzQ.png">
</picture>
<center>
<figcaption>Image from Chapter 1 slides of <a href="https://cloud.google.com/blog/products/gcp/learn-tensorflow-and-deep-learning-without-a-phd">"Learn TensorFlow and deep learning, without a Ph.D."</a> by Martin Görner. Cartoon images copyright: <a href="https://fr.123rf.com/profile_alexpokusay">alexpokusay /
123RF stock photos</a>. We tend to heavily rely on deep learning models for several
tasks, even for the simplest problems, but are we sure that we are given the
right answers?
</figcaption>
</center>
</figure>

Since the re-emergence of deep neural networks in 2012 by famously winning the
ImageNet Challenge (Krizhevsky et al., 2012), we have employed deep learning
models in a variety of real-world applications — to the point where we resort
to deep learning to solve even the simplest problems. Such applications range
from recommendation systems (Cheng et al., 2016) to medical diagnosis (Gulshan
et al., 2016). However, despite the state-of-the-art performance of deep
learning models in these specialized tasks, they are not infallible from
committing mistakes, in which the degree of seriousness of such mistakes vary
per application domain. So, the call for AI safety and trust is not surprising
(Lee & See, 2004; Varshney & Alemzadeh, 2017; Saria & Subbaswamy, 2019). For
years, much of the efforts were about improving the performance of models,
while further investigation on model limitations has not received an equal effort.

Despite receiving relatively less attention, there are some excellent works
on better understanding model predictions, and these include but are not
limited to the following: (a) the use of confidence calibration — where the
outputs of a classifier are transformed to values that can be interpreted
as probabilities (Platt, 1999; Zadrozny & Elkan, 2002; Guo et al., 2017),
(b) the use of ensemble networks to obtain confidence estimates
(Lakshminarayanan, Pritzel, & Blundell, 2017), and (c) using the softmax
probabilities of a model to identify misclassifications (Hendrycks &
Gimpel, 2016).

Now, the aforementioned methods use the reported score of a model for
confidence calibration — which may seem daunting even just to think about.
Enter: Trust Score. Instead of merely extending the said methods, Jiang et
al. (2018) developed an approach based on topological data analysis, where
they provide a single score for a prediction of a model, called trust
score.

Jibber-jabber aside, the trust score simply means the measurement of
agreement between a trained classifier f(x) and a modified nearest-neighbor
classifier g(x) on their prediction for test example x.
