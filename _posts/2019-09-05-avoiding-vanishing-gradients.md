---
layout: post
title: "Avoiding the vanishing gradients problem using gradient noise addition"
date: 2019-09-05
---

> We add random noise to a neural network's gradients as an attempt to avoid the vanishing gradients problem.

## Introduction

Neural networks are computational models used to approximate a function that models the relationship between the dataset features $x$ and labels $y$, i.e. $f(x) \approx y$. A neural net achieves this by learning the best parameters $\theta$ such that the difference between the prediction $f(x; \theta)$ and the label $y$ is minimal. They typically learn through gradient-based algorithms with the aid of backpropagation of errors observed at the output layer.

<figure>
<picture>
<img src="../../../images/vanishing-gradients-dnn.png">
</picture>
<figcaption>Illustrated using NN-SVG. A feed-forward neural network with two hidden layers. It learns to approximate the target label $y$ by learning the appropriate $\theta$ parameters with the criteria of minimizing the difference between its output label $f(x; \theta)$ and target label $y$.</figcaption>
</figure>

With this learning paradigm, neural nets have produced promising results in several tasks such as image classification (Krizhevsky et al. (2012); He et al. (2015)), image generation (Brock et al. (2018); Goodfellow et al. (2014); Radford et al. (2015); Zhu et al. (2017)), language modelling (Devlin et al. (2018); Howard and Ruder (2018)), audio synthesis (Engel et al. (2019); Oord et al. (2016)), and image captioning (Vinyals et al. (2015); Xu et al. (2015)) among others. However, this dominance was not always the case for neural nets. Before its resurgence in 2012 by winning the ImageNet Challenge, training neural networks was notoriously difficult.

The difficulty was due to a number of issues, e.g. the compute power and data were not sufficient to harness the full potential of neural nets. To a large extent, this is because neural nets were sensitive to initial weights (Glorot and Bengio, 2010), and they tended to prematurely stop learning as gradient values decrease to infinitesimally small values (Hochreiter et al., 2001) due to any or both of the following reasons: (1) their activation functions have small ranges of gradient values, and (2) their depth. This phenomenon is called the vanishing gradients problem.

The first reason of the problem mentioned above is our focus on this article. To reiterate in a different phrase, the vanishing gradient problem occurs when we train deep neural nets with gradient-based algorithms and backpropagation, where the gradients backpropagated through each hidden layer decrease to infinitesimally small values that the information necessary for the model to learn ceases to exist.

Naturally, a number of solutions have been proposed to alleviate this problem, e.g. the use of different activation functions (Nair and Hinton, 2010), and the use of residual connections (He et al., 2016). In this article, we review a number of proposed solutions to the vanishing gradients problem in the form of activation functions, but we limit our architecture to a feed-forward neural network. We will also look at an experimental method that could help avoid the vanishing gradients problem, and that could also help them converge faster and better.

## Gradient Noise Addition with Batch Normalization

Several research works have been proposed to solve the vanishing gradients problem, and such works include but are not limited to the introduction of new activation functions, and new architectures.

The simplest neural net architecture would consist of hidden layers with a logistic activation function that is trained with a gradient-based learning algorithm and backpropagation. The problem with such an architecture is its activation function that squashes the hidden layer values to $[0, 1] \in \mathbb{R}$. The gradients backpropagated with this function have a maximum value of 0.25 (see the Table 1), thus the values get saturated until there is no more useful information for the learning algorithm to use on updating the weights of the model.


|Activation|Function|Derivative|Min|Max|
|----------|--------|----------|---|---|
|Logistic|$\sigma(z) = \frac{1}{1+e^{-z}}$|$\sigma(z)(1 - \sigma(z))$|$y \rightarrow 0$|0.25|
|Hyperbolic Tangent|$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$|$1 - \left(\tanh\left(z \right) \right)^{2}$|$y \rightarrow 0$|1|
|ReLU|$\max(0, z)$|$1\ \text{if}\ z >= 0\ \text{else}\ 0$|0|1|
|Leaky ReLU|$\max(0.2z, z)$|$1\ \text{if}\ z >= 0\ \text{else}\ 0.2$|0|1|
|Swish|$f(z) = z \cdot \sigma(z)$|$f(z) + \sigma(z)(1 - f(z))$|$\approx-0.099$|$\approx1.0998$|

**Table 1. Activation functions used in neural nets, together with their respective derivatives, and maximum derivative values**

For years, the prevailing solution for this problem was the use of hyperbolic tangent, having a maximum gradient value of 1 (see Table 1). However, the gradient values still saturate with this function, which may be visually verified as seen in Figure 1. Hence, the rectified linear units (ReLU) activation function was introduced (Nair and Hinton, 2010).

The ReLU activation function has the same maximum gradient value of 1, but its advantage over logistic and hyperbolic tangent functions is that its activation values do not get saturated. However, ReLU has its own disadvantage, i.e. with its minimum gradient value of 0, it triggers the problem of “dead neurons”, i.e. a neuron has no activation value. So, yes, even though it avoids the saturation for non-negative values, its negative values would trigger the dead neurons phenomenon.

Due to this drawback, among the variations of ReLU that was developed was Leaky ReLU, which has a simple modification of having a lower bound slightly higher than zero. In turn, this modification allows models to avoid both the saturating gradient values and the “dead neurons” problem.

<figure>
<picture>
<img src="../../../images/vanishing-gradients-activation-functions.png">
</picture>
<figcaption>Figure 1. Plotted using matplotlib. Activation function values, their gradients, and their noise-augmented gradient values. For instance, adding Gaussian noise to the gradients of the logistic activation function increases its maximum value, i.e. from 0.25 to approximately 1.047831 (from a Gaussian distribution having a mean value of 0 and a standard deviation value of 0.5).</figcaption>
</figure>

But despite this modification, Ramachandran et al. (2017) claimed to have developed an even better function than ReLU, the “Swish” activation function. The aforementioned function could be described as a logistic-weighted linear function — it has a maximum gradient value of ~1.0998 (as it can also be seen in Table 1), and was found to outperform ReLU on CIFAR dataset (using ResNet), ImageNet dataset (using Inception and MobileNet), and machine translation (using a 12-layer Transformer model).

While these solutions focused more on formulating a new activation to improve the learning of a neural network, the work of Neelakantan et al. (2015) introduced a simple but effective approach of improving the neural network performance. The approach was simply to add gradient noise to improve the learning of a very deep neural network (see Eq. 1). Not only does it improve the performance of a neural network, but it also helps to avoid the problem of overfitting. Although the authors did not explicitly state that their proposed solution was designed to alleviate the vanishing gradients problem, it could be seen this way since the gradients computed during training are inflated — thus helping to avoid saturated gradient values which lead to the vanishing gradients problem.

\begin{equation}
\nabla_{\theta_t} J := \nabla_{\theta_t} J + \mathcal{N}(0, \sigma_t^2)
\end{equation}

The standard deviation $\sigma$ at time step t is then iteratively annealed by the following equation,

\begin{equation}
\sigma_{t}^2 := \dfrac{\eta = 1}{(1 + t)^{\gamma=0.55}}
\end{equation}

In the original paper by Neelakantan et al. (2015), the $\eta$ parameter was chosen from {0.1, 1.0}, while the $\gamma$ parameter was set to 0.55 in all their experiments.

TensorFlow 2.0 was used to implement the models and its computations for the experiments in this article. To implement the annealing gradient noise addition, we simply augment the gradients computed using tf.GradientTape by adding values from the Gaussian distribution generated with Eq. 1 and Eq. 2. That is, using tf.add as it can be seen in Line 7 from Snippet 1.

```python
def train_step(model, loss, features, labels, epoch):
  with tf.GradientTape() as tape:
    logits = model(features)
    train_loss = loss(logits, labels)
  gradients = tape.gradient(train_loss, model.trainable_variables)
  stddev = 1 / ((1 + epoch)**0.55)
  gradients = [tf.add(gradient, tf.random.normal(stddev=stddev, mean=0., shape=gradient.shape)) for gradient in gradients]
  model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  return train_loss, gradients
```

Finally, this approach was further augmented with the use of Batch Normalization (Ioffe and Szegedy, 2015). Thus, with this approach, the layer activations would be forced to take on unit Gaussian distribution during the beginning of a training session.

## Empirical Results
In the experiments that follow, the MNIST handwritten digits classification dataset (LeCun, Cortes, and Burges, 2010) was used for training and evaluating our neural networks. Each image was reshaped to a 784-dimensional vector, and then normalized by dividing each pixel value with the maximum pixel value (i.e. 255), and added random noise from a Gaussian distribution with a standard deviation of 5e-2 to elevate the difficulty to converge on the dataset.

### Improving Gradient Values
During training, we can observe the gradient distributions of a neural network as it learns. In Figure 2, we take a neural network with logistic activation function as an example. Since the logistic activation function has the least maximum gradient value (i.e. 0.25), we can consider observing the changes in its gradient distribution to be noteworthy.

<figure>
<picture>
<img src="https://miro.medium.com/max/875/1*e_Iw6AxcCyTwAx2sUKRjig.png">
</picture>
<figcaption>Figure 2. Figure from TensorBoard. Gradient distribution over time of neural nets with logistic activation function on the MNIST dataset. Top to bottom: Baseline model, experimental model with GNA, and experimental model with GNA + batch normalization.</figcaption>
</figure>

As we can see from the graphs above, the gradient distributions of the model drastically change from the baseline configuration to the experimental configurations, i.e. from considerably small values (+/- 4e-3 for two-layered neural network, and +/- 5e-6 for five-layered neural network) to relatively large values (+/- 4 for both two-layered and five-layered neural networks). While this does not guarantee superior model performance, it does give us an insight that there would be sufficient gradient values to propagate within the neural network, thus avoiding vanishing gradients. We turn to the next subsection to examine the classification performance of the models.

### Classification Performance
The models were trained using stochastic gradient descent (SGD) with Momentum (learning rate $\alpha = 3e-4$, momentum $\gamma = 9e-1$) on the perturbed MNIST dataset for 100 epochs, with mini-batch size of 1024 (for the two-layered neural network), and mini-batch size of 512 (for the five-layered neural network). Our networks consist of (1) two hidden layers with 512 neurons each, and (2) five hidden layers with the following neurons per hidden layer: 512, 512, 256, 256, 128. The weights for both the network architectures were initialized with Xavier initialization.

We can observe in Figures 3–6 that using the experimental methods, Gradient Noise Addition (GNA) and GNA with Batch Normalization (BN), help the neural net to converge faster and better in terms of loss and accuracy.

<figure>
<picture>
<img src="https://miro.medium.com/max/875/1*jYICNtXB0dWPzrx3l83Frw.png">
</picture>
<figcaption>Figure 3. Plotted using matplotlib. Training loss over time of the baseline and experimental (with GNA, and GNA + batch normalization) two-layered neural networks on the MNIST dataset.</figcaption>
</figure>


<figure>
<picture>
<img src="https://miro.medium.com/max/875/1*vtcpL0uGz51BuzRY76FBig.png">
</picture>
<figcaption>Figure 4. Plotted using matplotlib. Training accuracy over time of the baseline and experimental (with GNA, and GNA + batch normalization) two-layered neural networks on the MNIST dataset.</figcaption>
</figure>

<figure>
<picture>
<img src="https://miro.medium.com/max/875/1*rl7sVPtB7unZyyv-iECCzg.png">
</picture>
<figcaption>Table 2. Test accuracy of the baseline and experimental (with GNA, and GNA + batch normalization) two-layered neural networks on the MNIST dataset.</figcaption>
</figure>

From Table 2, we can see the test accuracy values using GNA and GNA+BN drastically improved — most notably on the logistic-based neural network.

<figure>
<picture>
<img src="https://miro.medium.com/max/875/1*dEH9nZZGjT2sI0e8QOe7Jw.png">
</picture>
<figcaption>Figure 5. Plotted using matplotlib. Training loss over time of the baseline and experimental (with GNA, and GNA + batch normalization) five-layered neural networks on the MNIST dataset.</figcaption>
</figure>

<figure>
<picture>
<img src="https://miro.medium.com/max/875/1*jHALiJAWtZCNN0x1tE10sA.png">
</picture>
<figcaption>Figure 6. Plotted using matplotlib. Training accuracy over time of the baseline and experimental (with GNA, and GNA + batch normalization) five-layered neural networks on the MNIST dataset.</figcaption>
</figure>

<figure>
<picture>
<img src="https://miro.medium.com/max/875/1*vU29lPlif7LYg9gJ3NJtCg.png">
</picture>
<figcaption>Table 3. Test accuracy of the baseline and experimental (with GNA, and GNA + batch normalization) five-layered neural networks on the MNIST dataset.</figcaption>
</figure>

We have seen that all the baseline two-layered neural networks improved with GNA and GNA+BN from the results tallied in Table 2. However, for the five-layered neural networks (see Table 3), the ReLU-based model failed to improve in terms of test accuracy. Furthermore, we can see that the TanH-based neural network had a better test accuracy in this configuration than the ReLU-based model. We can attribute this to the fact that we used Xavier initialization (in which TanH performs optimally) rather than He initialization (in which ReLU performs optimally).


In general, these results on the five-layered neural network support the statement earlier that the inflation of gradient values do not necessarily guarantee superior performance.

But what’s interesting here is the drastic improvement of the baseline model with Swish activation function — an increase in test accuracy as high as 54.47%.

Despite the results for the two-layered neural network where the Swish-based model had slightly lower test accuracies than the ReLU- and Leaky RELU-based ones, we can see that for the five-layered neural network, the Swish-based model had a higher test accuracy than the ReLU-based models (but slightly lower than the Leaky ReLU-based model). This somehow corroborates the fact the Swish outperforms ReLU on deeper networks as exhibited by Ramachandran et al. (2017) on their results for their 12-layer Transformer model.


## Closing Remarks
In this article, we had an overview of the gradient values propagated in a neural network with respect to a set of activation functions (logistic, hyperbolic tangent, rectified linear units, low-bounded rectified linear units, and logistic weighted linear units). We used an approach combining gradient noise addition and batch normalization as an attempt to alleviate the vanishing gradients problem. We saw that the model performance improved as high as 54.47% in terms of test accuracy. Furthermore, with the experimental methods, the models converged faster and better compared to their baseline counterparts.

Our exploration in this article is only the tip of the iceberg as there are a lot of things that can be discussed regarding the vanishing gradients problem. For instance, how about if we tried both Xavier and He initialization schemes, and compared them with zero and random initialization for the baseline and experimental regimes? To what extent does the gradient noise addition help? That is, can it still help a neural network with 30 layers or more? What if we also used layer normalization or weight normalization? How will it fair against or with residual networks?

I hope we have covered enough in this article to make you wonder more about the vanishing gradients problem and about the different ways to help avoid it.

The full code is available [here](https://github.com/afagarap/vanishing-gradients). In case you have any feedback, you may reach me through [Twitter](https://twitter.com/afagarap). We can also connect through [LinkedIn](https://linkedin.com/in/abienfredagarap)!

If you enjoyed reading this article, perhaps you will also find my blog on [Implementing an Autoencoder in TensorFlow 2.0](https://afagarap.github.io/2019/03/20/implementing-autoencoder-in-tensorflow-2.0.html) interesting!

### References

* Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. “Imagenet classification with deep convolutional neural networks.” Advances in neural information processing systems. 2012.
* He, Kaiming, et al. “Deep residual learning for image recognition.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
* Brock, Andrew, Jeff Donahue, and Karen Simonyan. “Large scale gan training for high fidelity natural image synthesis.” arXiv preprint arXiv:1809.11096 (2018).
* Goodfellow, Ian, et al. “Generative adversarial nets.” Advances in neural information processing systems. 2014.
* Radford, Alec, Luke Metz, and Soumith Chintala. “Unsupervised representation learning with deep convolutional generative adversarial networks.” arXiv preprint arXiv:1511.06434 (2015).
* Zhu, Jun-Yan, et al. “Unpaired image-to-image translation using cycle-consistent adversarial networks.” Proceedings of the IEEE international conference on computer vision. 2017.
* Devlin, Jacob, et al. “Bert: Pre-training of deep bidirectional transformers for language understanding.” arXiv preprint arXiv:1810.04805 (2018).
* Howard, Jeremy, and Sebastian Ruder. “Universal language model fine-tuning for text classification.” arXiv preprint arXiv:1801.06146 (2018).
* Engel, Jesse, et al. “Gansynth: Adversarial neural audio synthesis.” arXiv preprint arXiv:1902.08710 (2019).
* Oord, Aaron van den, et al. “Wavenet: A generative model for raw audio.” arXiv preprint arXiv:1609.03499 (2016).
* Vinyals, Oriol, et al. “Show and tell: A neural image caption generator.” Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
* Xu, Kelvin, et al. “Show, attend and tell: Neural image caption generation with visual attention.” International conference on machine learning. 2015.
* Glorot, Xavier, and Yoshua Bengio. “Understanding the difficulty of training deep feedforward neural networks.” Proceedings of the thirteenth international conference on artificial intelligence and statistics. 2010.
* Hochreiter, Sepp, et al. “Gradient flow in recurrent nets: the difficulty of learning long-term dependencies.” (2001).
* Nair, Vinod, and Geoffrey E. Hinton. “Rectified linear units improve restricted boltzmann machines.” Proceedings of the 27th international conference on machine learning (ICML-10). 2010.
* Ramachandran, Prajit, Barret Zoph, and Quoc V. Le. “Swish: a self-gated activation function.” arXiv preprint arXiv:1710.05941 7 (2017).
* Neelakantan, Arvind, et al. “Adding gradient noise improves learning for very deep networks.” arXiv preprint arXiv:1511.06807 (2015).
* Ioffe, Sergey, and Christian Szegedy. “Batch normalization: Accelerating deep network training by reducing internal covariate shift.” arXiv preprint arXiv:1502.03167 (2015).
* LeCun, Yann, Corinna Cortes, and C. J. Burges. “MNIST handwritten digit database.” AT&T Labs [Online]. Available: http://yann.lecun.com/exdb/mnist (2010): 18.
