---
layout: single
title:  "word2vec之一：Negative Sampling的目标函数推导"
date:   2023-06-12 15:00:00 +0800
categories: [机器学习]
tags: [word2vec, embedding]
render_with_liquid: false
---

>内容提要：
>1. word2vec用向量点积来衡量词的相似度
>2. softmax难以计算，通过negative sampling方法近似计算，背后的理论基础是NCE
>3. negative sampling的目标函数跟LR一样，相比较理论上NCE还是做了一些近似

人类对一个词的含义会有多个维度的理解。说到“中国”这个词，人类脑海中会闪现一些概念，比如：几千年的文明古国，语言是汉语，位于亚洲，黄种人。这些概念拼在一起就是对“中国”很好的理解。

计算机是如何理解一个词的呢？为了模仿人类，给定一个词$w$，计算机用一个d维向量$\vec w=(w_1,w_2,\dots,w_d)$来表示一个词，d的大小一般100~1000，向量中的每一个维度$w_N$都是这个词某方面的描述。举个直观的例子，假设第一个维度$w_1$是描述一个国家的历史久远程度，那么“中国”这个词的向量第一个维度$w_1$就会很大，“美国”的这个维度就小一些。然而，假设归假设，实际上向量的各个维度的含义其实是隐含的，只知道表示了某种含义，但具体是什么含义是未知的。

有很多方法可以得到一个词的向量表示（embedding），其中最火爆的当属2013年Google发布的word2vec。本文会简单回顾下word2vec模型，重点介绍word2vec目标函数的演化过程

PS：[ref][/ref]中的文本表示文献引用，[footnote][/footnote]中的文本表示注解

<h1> word2vec回顾 </h1>

参考[ref]word2vec: Tomas Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality, NIPS 13[/ref]

我们的任务是求解词向量，那么，如何评价词向量的好坏呢？word2vec有两个model形式，一个是CBOW，一个是Skip-gram。CBOW认为好的词向量可以使我们通过context来预测当前的词，Skip-gram认为好的词向量可以使我们通过当前的词来预测context。模型的形式如下图所示。目前来看Skip-gram的效果要由于CBOW。本文只讲Skip-gram，不仅是因为它的效果好，更是因为下一篇文章会讲到word2vec与矩阵分解等价关系，等价的就是Skip-gram的Negative Sampling方法。

<h2> Skip-gram的softmax目标函数 </h2>

Skip-gram的目标是找到一个词合适的向量表示用来预测周围的词，形式化的说，给定训练语料，即一个词序列$w_1,w_2,\dots,w_T$，长度为$T$，Skip-gram模型的目标最大化平均的对数概率

\begin{equation} \label{eq:objective}
\frac{1}{T}\sum_{t=1}^T \sum_{-l \leq j \leq l,j\ne0}{\log p(w_{t+j} \vert w_t)}
\end{equation}

其中，$l$是上下文窗口的大小。基础的Skip-gram模型通过一个softmax来定义概率分布函数$p(w_{t+j} \vert w_t)$

\begin{equation} \label{eq:softmax}
p(c \vert w)=\frac{\exp(\vec{w}\cdot\vec{c})}{\sum_{c'\in V_C} {\exp(\vec{w}\cdot\vec{c'})}}
\end{equation}

其中，$w$是当前词，$c$是窗口内上下文的某个词，$\vec w \in\mathbb{R}^d$是$w$的向量表示，也就是词向量，$\vec c \in\mathbb{R}^d$是$c$的向量表示，$d$是词向量维度。这里注意，同一个词作为当前词和上下文词时候的词向量是不同的，它们属于两个空间。$\vec{w}\cdot\vec{c}$可以理解为当前词和上下文的相关性，softmax把相关性转化成了概率。$V_W$是当前词的词表，$V_C$是上下文词的词表，对于word2vec，$V_W$和$V_C$是相同的。这里用两个符号表示，是因为他们对应的词向量是不同的。

<h2> softmax存在的问题 </h2>

词表$V_C$的大小一般几十万到上百万，导致这个softmax公式计算复杂度太高，很难落地。显然，梯度$\nabla \log p(c \vert w)$的计算量跟$ \vert V_W \vert $成正比。所以，需要找到近似计算softmax的方法。论文提出了下面两种方法

<ol>
     <li>Hierarchical Softmax</li>
     <li>Negative Sampling</li>
</ol>

上面两种方法可以代表两类近似softmax的方法，一类是softmax based，保留了softmax层，但是修改了结构来提高计算效率，比如Hierarchical Softmax；另外一类是sample based，这一类方法抛弃了softmax，寻求其他的目标函数来近似softmax。

Hierarchical Softmax不是本文重点，略去不讲。本文只讲Negative Sampling方法，这个方法背后的理论基础是Noise Contrastive Estimation（NCE）。跟最大似然估计（MLE）一样，NCE也是用来估计概率分布函数的，但是NCE采用基于分类的方法来做估计，提出的目标函数的计算复杂度相比softmax小很多，可以简单理解为把概率分布估计问题转化成了分类问题。Negative Sampling方法的单个样本的目标函数如下。

\begin{equation} \label{eq:ns}
\log\sigma(\vec{w}\cdot\vec{c})+k \mathbb{E}_{c_N \sim P_n(c)}\left[\log\sigma(-\vec{c_N}\cdot\vec{w})\right]
\end{equation}

下一小节会介绍NCE，并且解释为什么Negative Sampling方法目标函数可以这样写。

<h1> 概率分布估计问题 </h1>

这里考虑离散变量的概率分布。

回顾一下概率分布估计问题，一般是这样的图景。现实世界中，我们观测到了一些数据，$X=(x_1,\dots,{x}_{T_d})$，${x}\in\mathbb{R}^n$服从一个未知的概率分布函数（pdf）$p_d$。因为我们观测到的数据只是全部数据（population）中的一个子集，所以这个真实的概率分布函数我们一般无法知道。这时，我们会猜测一个pdf $p_m(.;\theta)$来拟合真实的$p_d$，函数里面包含一些参数$\theta$，目标是通过已观测数据，求解最优的参数${\theta^\ast}$，以便我们猜测的这个概率分布函数更好的拟合已观测到的数据，即$p_d(.)=p_m(.;{\theta^*})$。

举个具体的例子，对于word2vec来说，词序列中上下文窗口内的一个个pair $(w,c)$就是已观测的数据${x}$，公式（\ref{eq:softmax}）中的softmax就是我们猜测的概率分布函数$p_m$，公式（\ref{eq:objective}）是通过MLE构造的目标函数，词向量就是需要求解的参数$\theta$，最终求得的词向量是${\theta^*}$。

通过上面形式化的定义，我们把概率分布估计问题转化为求解${\theta^*}$。同时，求解的过程中还有2个重要的限制条件，即任何求解出来的${\hat\theta}$所确定的$p_m(.,{\hat\theta})$都需要归一化并且>=0

\begin{equation} \label{eq:norm}
\sum{p_m(.;{\hat\theta})}=1, \qquad p_m(.;{\hat\theta})\geq 0
\end{equation}

这两个条件一定要满足，这样$p_m(.,{\hat\theta})$才是概率，才可以用MLE来求解${\theta^*}$。然而，现实中，有些问题分布函数不是天然归一化的，用$p^0_m(.;{\alpha})$表示[footnote]$p^0_m$的形式一般都是先定义一个跟待求解的问题相关的函数，用$f({\alpha})$表示，其中${\alpha}$是参数。，然后加个e的指数$\exp(f({\alpha}))$[/footnote]。虽然$p^0_m$不是天然归一化的，但是归一化可以做，可以利用一个partition function $Z({\alpha})$来做归一化

\begin{equation} \label{eq:Z}
Z({\alpha})=\sum_{u} p^0_m(u;{\alpha})
\end{equation}

这样，$p^0_m(.;{\alpha})/Z({\alpha})$就是归一化的了。一般吉布斯分布（Gibbs distributions）、马尔可夫网络（Markov networks）和多层神经网络（multilayer networks）会采用这样的方式。然而，现实中这些问题的$Z({\alpha})$难以计算，一般都没有解析形式（closed form），计算复杂度高。

在这种情况下，NCE出现了，NCE就是来解决这种归一化难以计算的问题的。

<h1> Noise Contrastive Estimation </h1>

参考[ref]nce: Michael U. Gutmann et al., Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics, JMLR 12[/ref]

<h2> 简介 </h2>

NCE和MLE一样，都是估计参数的方法。不同的是NCE有个MLE没有的优点，就是可以处理概率分布函数很难归一化的情况（我们姑且还叫它概率分布函数）。NCE有两个特点

<ol>
     <li>把归一化项整体看成一个普通的参数，跟原有参数合并到一起估计</li>
     <li>把概率分布估计问题转化成了分类问题，提出了个基于逻辑回归（LR）的目标函数</li>
</ol>

<h2> 把归一化项看做一个参数 </h2>

先说第一个特点。

归一化项都不用计算了，直接是个参数，你说是不是减少了很多计算。具体是怎么做的呢？令$Z$，或者$c=\log 1/Z$不再是${\alpha}$的函数，而是一个普通的要求解的参数，则

$$
\begin{equation} \label{eq:zc}
\begin{split}
\log p_m(.;{\theta})&=\log \frac{p^0_m(.;{\alpha})}{Z({\alpha})}\\
                       &=\log {p^0_m(.;{\alpha})}+c
\end{split}
\end{equation}
$$

其中参数$\theta=(\alpha,c)$。对于其中的一个解${\hat\theta}=({\hat\alpha},\hat{c})$，可以这样直观的理解，$p^0_m(.;{\hat\alpha})$的几何形状跟$p_d$类似，只是通过$\hat{c}$做了些缩放。

举个具体的例子，词序列中上下文窗口内的一个个pair $(w,c)$服从一个概率分布$p_d$，这个真实的分布我们无法知道，因为毕竟观测到的词序列只是这个世界上全部文本的一个子集。我们猜测一个概率分布函数$p_m$用来拟合$p_d$，也就那个softmax公式（\ref{eq:softmax}）定义的$p(c \vert w)$。其中，$\vec{w}\cdot\vec{c}$可以理解为当前词和上下文的相关性，通过它构造的$\exp(\vec{w}\cdot\vec{c})$就是那个未归一化的分布$p^0_m(.;{\alpha})$，参数${\alpha}$就是我们要求解的词向量，分母就是那个partition function $Z({\alpha})$。因为$Z$很难计算，所以NCE把$Z$看成一个参数来估计。

<h2> 通过比较来估计概率分布 </h2>

Density Estimation by Comparison

<h3> 直观理解 </h3>

有时，单独的描述一个东西可能很难。比如，要你描述一个人有多好，直接描述就很难，不过相对描述就容易的多——比A差点、比B好点。

机器学习分为有监督学习和无监督学习。无监督学习是指给定观测数据，目的学习到数据的内部结构和数据之间的关系，概率分布估计问题属于无监督学习。有监督学习除了给定数据，还会给数据的标记，目的是学习到一个好的分类器。所以，对于求解词向量这个问题，上面的softmax是无监督学习。

那么，是否可以换个角度，把这个问题看成一个有监督的学习问题？有监督与非监督的不同，在于看待数据的方式上。首先要明确一下，对于word2vec，给定了一个词序列$w_1,w_2,\dots,w_T$，上下文窗口大小为$l$，观测到的数据到是上下文窗口内的一个个的pair $(w,c)$。这样的话，就有两种角度来看待数据。（1）如果把数据只是看成是一个个的pair $(w,c)$，估计这些pair的概率分布[footnote]softmax虽然定义的是条件概率$p(c \vert w)$，但是根据贝叶斯公式，$p(w,c)=p(c \vert w)p(w)$，$p(w)$可以一般是已知的，可以认为softmax的目的就是估计pair$(w,c)$的概率分布[/footnote]，这是无监督的角度；（2）如果在观测到的一个个pair $(w,c)$基础上，给它们加个标记为1，再加上采样的一些上下文窗口中未观测到的pair $(w,c_N)$，标记为0，学习一个分类器，这就是有监督的角度。所以，有监督和无监督内在是有某种内在联系的。NCE的出发点就是这样，基于这种内在联系，把概率分布估计问题转化了成了一个分类问题。

采样的未观测到的数据就是噪声（noise），NCE的名字也来自于此。

<h3> 形式化定义 </h3>

给定了已观测的数据$X=(x_1,\dots,x_{T_d})$，${x}\in\mathbb{R}^n$，NCE引入了另一份数据$Y$，即噪声（Noise），采用相对描述的方式描述$X$。假设$Y$是i.i.d的样本集合$(y_1,\dots,y_{T_n})$，${y}\in\mathbb{R}^n$，概率分布函数是$p_n$，下标n的含义是noise。两个概率分布函数的比例$p_d/p_n$就构成了$X$的一个相对的描述，如果$p_n$是已知的，我们就可以从这个比例里面得到$p_d$。也就是说，如果我们知道了$X$和$Y$之间的区别，并且知道$Y$的属性，那么就可以推断出$X$的属性。

两个数据集的比较可以通过分类来实现。下面，我们可以看到基于逻辑回归（LR）训练一个分类器，可以得到$X$的一个相对描述$p_d/p_n$

<h3> NCE目标函数的推导 </h3>

令$U=(u_1,\dots,u_{T_d+T_n})$表示$X$和$Y$的并集，并且赋予$u_t$的一个类别标记$C_t$：$C_t=1$ if $u_t \in X$，$C_t=0$ if $u_t \in Y$。LR的核心是估计给定数据的条件下类别标记的后验概率$P(C=1 \vert  u;\theta)$，下面就推导一下这个后验概率的形式。$p(. \vert C=1)$是给定类别的条件下数据的分布，由于真实的数据分布$p_d$我们不知道，所以用$p_m( u \vert C=1,\theta)$表示$P(. \vert C=1)$。那么给定类目标记，数据的分布：

\begin{equation} \label{eq:give_class}
P( u \vert C=1,\theta)=p_m( u,{\theta}), \qquad\qquad P( u \vert C=0)=p_n( u)
\end{equation}

类目标记的先验概率：$p(C=1)=T_d/(T_d+T_n)$，$p(C=0)=T_n/(T_d+T_n)$，令$v=p(C=0)/p(C=1)=T_n/T_d$，表示噪声是数据的倍数，则类别标记的后验概率如下，其中第一步根据贝叶斯公式展开。

$$
\begin{equation} 
\begin{split}
P(C=1 \vert u;\theta)&=\frac{P( u \vert C=1;\theta)P(C=1)}{P( u \vert C=1;\theta)P(C=1)+P( u \vert C=0;\theta)P(C=0)}\\
&=\frac{P( u \vert C=1;\theta)}{P( u \vert C=1;\theta)+\frac{P(C=0)}{P(C=1)}P( u \vert C=0;\theta)}\\
&=\frac{p_m( u;{\theta})}{p_m( u;{\theta})+vp_n( u)}\\
&=\frac{1}{1+v\frac{p_n( u)}{p_m( u;{\theta})}}\\
&=\frac{1}{1+v\exp[-\log\frac{p_m( u;{\theta})}{p_n( u)}]}\\
&=h( u;\theta)\\
P(C=0 \vert  u;\theta)&=1-h( u;\theta)
\end{split}
\label{eq:like_lr}
\end{equation}
$$

令$G( u;{\theta})$表示$p_m( u;{\theta})$和$p_n( u)$的对数比例，即


\begin{equation} \label{eq:log_ratio}
G( u;{\theta})=\log\frac{p_m( u;{\theta})}{p_n( u)}=\log p_m( u;{\theta})-\log p_n( u)
\end{equation}

\begin{equation} \label{eq:h_sigma}
h( u;\theta)=\sigma_v(G( u;{\theta}))=\frac{1}{1+v\exp[-G( u;{\theta})]}
\end{equation}

条件对数似然如下：

$$
\begin{equation} 
\begin{split}
\ell(\theta)&=\frac{1}{T_d}(\sum_{t=1}^{T_d+T_n} C_t\log P(C_t=1 \vert u_t;\theta)+(1-C_t)\log P(C_t=0 \vert u_t;\theta))\\
&=\frac{1}{T_d}(\sum_{t=1}^{T_d} \log[h({x}_t;\theta)] + \sum_{t=1}^{T_n} \log[1-h({y}_t;\theta)])\\
&=\frac{1}{T_d}\sum_{t=1}^{T_d} \log[h({x}_t;\theta)] + v\frac{1}{T_n}\sum_{t=1}^{T_n} \log[1-h({y}_t;\theta)]\\
\end{split}
\label{eq:nce_loss}
\end{equation}
$$

通过优化$\ell(\theta)$来求解${\theta}$，可以得到一个$G(u;{\hat\theta})$的一个估计，即$\log(p_d/p_n)$，进而可以得到$X$相对于$Y$的一个描述。至此，我们成功的把概率分布估计问题转化成了基于LR的分类问题，也就是无监督的问题转化成了有监督的问题。

假设我们知道噪声的分布$p_n$，并且假设噪声的数量足够多，根据大数定理，公式（\ref{eq:nce_loss}）还可以写成

$$
\begin{equation} \label{eq:like_word2vec}
\begin{split}
\ell(\theta)&=\frac{1}{T_d}\sum_{t=1}^{T_d} \log[h({x}_t;\theta)] + v\mathbb{E}_{{y_t} \sim P_n(y)} \log[1-h({y}_t;\theta)]\\
&=\frac{1}{T_d}\sum_{t=1}^{T_d} \log[h({x}_t;\theta)] + v\frac{1}{T_d}\sum_{t=1}^{T_d}\mathbb{E}_{{y_t} \sim P_n(y)} \log[1-h({y}_t;\theta)]\\
&=\frac{1}{T_d}\sum_{t=1}^{T_d} (\log[h({x}_t;\theta)] + v\mathbb{E}_{{y_t} \sim P_n(y)} \log[1-h({y}_t;\theta)])\\
\end{split}
\end{equation}
$$

个人理解，softmax和nce的区别只是在于利用“噪声”的方式不同。softmax把噪声放在了分母里面，用于计算概率；nce把噪声跟数据做对比。

<h1> Negative Sampling与NCE </h1>

<h2> 通过目标函数理解word2vec </h2>

回顾一下Negative Sampling单个pair的目标函数

\begin{equation} \label{eq:ns2}
\log\sigma(\vec{w}\cdot\vec{c})+k \mathbb{E}_{c_N \sim P_n(c)}\left[\log\sigma(-\vec{c_N}\cdot\vec{w})\right]
\end{equation}

从原始的softmax出发，基于NCE的目标函数，构造word2vec单个pair对应的目标函数，结合（\ref{eq:like_lr}）（\ref{eq:log_ratio}）（\ref{eq:h_sigma}）一起推导

$$
\begin{equation} \label{eq:w2v_nce}
\begin{split}
p_m( u;{\theta})&=p_m(w,c;{\theta}) \\
                      &=p_m(c \vert w;{\theta})p(w) \\
                      \\
G( u;{\theta})  &=G(w,c;{\theta}) \\
                      &=\log p_m(w,c;{\theta})-\log p_n(c)\\
                      &=\log p_m(c \vert w;{\theta}) + \log p(w) -\log p_n(c)\\
                      &=\log \frac{\exp(\vec{w}\cdot\vec{c})}{Z} + \log p(w) -\log p_n(c)\\
                      &=\vec{w}\cdot\vec{c} -\log Z + \log p(w) -\log p_n(c)\\
                      \\
\log[h({x}_t;\theta)] + v\mathbb{E}_{{y_t} \sim P_n(y)}\log[1-h({y}_t;\theta)]
                      &=\log[\sigma_v(G(w,c;{\theta}))] + v\mathbb{E}_{{c_N} \sim p_n(c)}\log[1-\sigma_v(G(w,c_N;{\theta}))]\\
\end{split}
\end{equation}
$$

其中，$\theta=(\vec w,\vec c,Z)$是参数。对比\ref{eq:ns}可以知道，Negative Sampling做了几处简化：

<ol>
     <li>忽略了跟参数$\theta$无关的$p(w),p_n(c)$[footnote]给定pair $(w,c)$，可以认为它们都是常数[/footnote]</li>
     <li>忽略了归一化项$Z$[footnote]忽略是忽略了，但不代表假设$\exp(\vec{w}\cdot\vec{c})$是自然归一化的，因为NCE被证明可以处理未归一化的情况，这部分暂时无力推导。。[/footnote]</li>
        <li>假设$\sigma_v$中的$v$为1，即假设$\sigma_v$是一个标准的sigmoid函数</li>
</ol>

根据（\ref{eq:like_lr}），对于LR那个核心的类别标记的后验分布，从NCE和Negative Sampling的角度看分别是：

$$
\begin{equation} \label{eq:lr_post}
\begin{split}
P_{NCE}(C=1 \vert w,c;\theta)&=\frac{1}{1+v\exp[-(\vec{w}\cdot\vec{c} -\log Z + \log p(w) -\log p_n(c))]}\\
P_{NEG}(C=1 \vert w,c;\theta)&=\frac{1}{1+\exp(-\vec{w}\cdot\vec{c})}\\
\end{split}
\end{equation}
$$

前文说过，$\vec{w}\cdot\vec{c}$表示词和上下文的相关度，公式（\ref{eq:log_ratio}）也印证了这一点。词和上下文越相关，$\vec{w}$和$\vec{c}$越相似，点乘结果越大，$\frac{p_m(u;\theta)}{p_n( u)}$比值越大，$(w,c)$越可能是真实的上下文；词和上下文越不相关，$\vec{w}$和$\vec{c}$越不相似，点乘结果越小，即$\frac{p_m(u;\theta)}{p_n(u)}$比值越小，$(w,c)$越可能是没什么关系。

对于word2vec，有着相似上下文的词会相似一些，但是这只是直观的看，它的目标函数并没有直接衡量两个词的相似度。

<h2> 近似计算期望 </h2>

参考[ref]Chris Dyer, Notes on Noise Contrastive Estimation and Negative Sampling, 14[/ref]

Negative Sampling的目标函数（\ref{eq:ns}）里面有个期望，如果计算真实的期望，计算量又会很大。所以，为了避免直接计算期望，实际上是用蒙特卡洛方法采样了k个噪声样本来近似计算期望

$$
\begin{equation} \label{eq:mc_e}
\begin{split}
\log\sigma(\vec{w}\cdot\vec{c})+ \sum_{i=1,c_N \sim P_n(c)}^k \left[\log\sigma(-\vec{c_N}\cdot\vec{w})\right]
\end{split}
\end{equation}
$$

<h2> 噪声分布 </h2>

为了做分类，NCE会根据分布$P_n(c)$采样一些噪声（Noise），跟已观测的到的数据做比较，$P_n(c)$是已知的。Negative Sampling中，$P_n(c)$从上下文词的unigram分布$U(c)$而来，作者经过多次实验，发现$U(c)^{3/4}/Z$的效果是最好的，其中$Z$是归一化参数。

<h1> 参考文献 </h1>

* word2vec: Tomas Mikolov et al., Distributed Representations of Words and Phrases and their Compositionality, NIPS 13 
* nce: Michael U. Gutmann et al., Noise-Contrastive Estimation of Unnormalized Statistical Models, with Applications to Natural Image Statistics, JMLR 12 
* Chris Dyer, Notes on Noise Contrastive Estimation and Negative Sampling, 14 