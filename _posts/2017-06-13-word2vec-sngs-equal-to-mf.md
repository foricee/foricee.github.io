---
title:  "word2vec之二：Negative Sampling与矩阵分解的等价关系"
date:   2017-06-13 15:00:00 +0800
tags:
  - 机器学习
  - word2vec 
  - embedding
---

>内容提要：
>1. SGNS相当于做矩阵分解，矩阵的元素是当前词和上下文词的互信息
>2. 互信息的主要部分是词的共现次数取log，剩下的部分是各个词自己的词频（Glove及其后面的方法把这部分常量当做bias去学了）

PS：[ref][/ref]中的文本表示文献引用，[footnote][/footnote]中的文本表示注解

想要求解词向量，word2vec是你很好的选择。像上一篇文章《<a href="/2017/06/12/word2vec-negative-sampling-objective-function.html">word2vec之一：Negative Sampling的目标函数推导</a>》定义的那样，给定一个词序列$w_1,w_2,\dots,w_T$，可以求解每个词$w$的词向量$\vec w\in\mathbb{R}^d$和每个上下文词的向量$\vec c\in\mathbb{R}^d$。把向量拼在一起，可以得到两个矩阵，词矩阵$W$和上下文矩阵$C$[footnote]word2vec只输出了$W$，忽略了$C$[/footnote]。出于直觉，考虑矩阵的乘积$W \times C^{\top}=M$，从这个角度看，word2vec像是在做一个隐式的矩阵分解（MF）。

那么，是在分解哪个矩阵呢？像上一篇文章里指出的那样，给定词序列，我们的训练数据其实是当前词和窗口内的上下文词组成的一个个pair $(w,c)$。所以，词和上下文的这种pair关系也可以用一个矩阵$M$描述，行表示当前词，列表示上下文词，矩阵的元素表示词和上下文之间的关联关系。按照上面矩阵乘积的定义，$W_i \cdot C_j = M_{ij}$。

把求解word2vec看成矩阵分解的意义在于，如果我们可以很方便的得到$M$，那么用矩阵分解的方法就可以求得词矩阵$W$了。也就是除了Negative Sampling，又多了一种方法求解词向量。那么，$M_{ij}$到底是什么呢？

本文主要内容是证明word2vec的Negative Sampling方法跟矩阵分解是等价的，最后介绍从矩阵分解的角度求解词向量的目标函数。

# Skip gram的Negative Sampling的方法 

Skip gram Negative Sampling简称SGNS。

## 符号定义 

$V_W$是当前词的词表，$V_C$是上下文词的词表，对于word2vec，$V_W$和$V_C$是相同的。这里用两个符号表示，是因为他们对应的词向量是不同的。这些词来源于一个语料$w_1,w_2,\dots,w_T$，$T$是语料长度，一般上亿。当前词$w_i$的上下文是它周围大小为$l$的窗口$w_{i-l},\dots,w_{i-1},w_{i+1},\dots,w_{i+l}$。当前词和窗口内上下文词组成的pair是$(w,c)$，$D$所有pair的集合。$\\#(w,c)$表示pair在$D$中出现的次数，$\\#(w)=\sum_{c' \in V_C} \\#(w,c')$和$\\#(c)=\sum_{w' \in V_C} \\#(w',c)$分别表示$w$和$c$在$D$中出现的次数。

$\vec w \in\mathbb{R}^d$是$w$的向量表示，也就是词向量，$\vec c \in\mathbb{R}^d$是$c$的向量表示，$d$是词向量维度。$\vec w$是大小为$ \vert V_W \vert \times d$的矩阵$W$的某一行，$\vec c$是大小为$ \vert V_C \vert \times d$的矩阵$C$的某一行。$W_i$（$C_i$）指的是第i个当前词（上下文词）的词向量。

## 目标函数 

回顾上一篇文章，给定一个pair $(w,c)$，类别标记的后验概率

$$
\begin{equation} \label{eq:lr_post}
\begin{split}
P(C=1 \vert w,c)&=\frac{1}{1+\exp(-\vec{w}\cdot\vec{c})}
\end{split}
\end{equation}
$$

这个pair $(w,c)$的目标函数

\begin{equation} \label{eq:ns}
\log\sigma(\vec{w}\cdot\vec{c})+k \mathbb{E}_{c_N \sim P_n(c)}\left[\log\sigma(-\vec{c_N}\cdot\vec{w})\right]
\end{equation}

$k$是采的负样本的数量，$c_N$是采样的上下文，服从unigram分布$P_n(c)=\frac{\\#(c)}{ \vert D \vert }$[footnote]准确的说，Negative Sampling还开了3/4次方，作者为了方便，直接使用unigram分布[/footnote]。

整体需要最大化的目标函数如下。优化这个目标函数会使观测到的词和上下文有着相似的词向量，未观测到的反之。

\begin{equation} 
\ell=\sum_{w \in V_W} \sum_{c \in V_C} \\#(w,c)(\log\sigma(\vec{w}\cdot\vec{c})+k \mathbb{E}_{c_N \sim P_n(w)}[\log\sigma(-\vec{c_N}\cdot\vec{w})])
\label{eq:full_obj}
\end{equation}

# SGNS所分解的矩阵 

参考[ref]Goldberg, Neural Word Embedding as Implicit Matrix Factorization, NIPS 14[/ref]

文章开始说到，如果我们知道了$M$，那么用矩阵分解方法就可以求得词向量。那么，$M$中的元素$M_{ij}$到底是什么呢？

$$
\begin{bmatrix}
\dots & \dots & \dots \\
\dots & M_{ij}=? & \dots \\
\dots & \dots & \dots \\
\end{bmatrix}_{M_{ \vert V_W \vert  \times  \vert V_C \vert }}
 =
\begin{bmatrix}
\vdots & \vdots & \vdots & \vdots \\
w_{i_1} & w_{i_2} & \dots & w_{i_d} \\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}_{W_{ \vert V_W \vert  \times d}}
\times
\begin{bmatrix}
\dots & c_{j_1} & \dots \\
\dots & c_{j_2} & \dots \\
\dots & \vdots & \dots \\
\dots & c_{j_d} & \dots
\end{bmatrix}_{C^{\top}_{ \vert V_C \vert  \times d}}
$$

推导之前，需要做一个假设，假设词向量的维度$d$足够大，分解得到的词向量矩阵$W$、$C$可以完全重建$M$，这样，每一个$\vec{w}\cdot\vec{c}$的点乘乘积可以认为独立于其他的点乘，也就是把$\vec{w}\cdot\vec{c}$整体看成是独立的。

开始推导，重写整体目标公式(\ref{eq:full_obj})，构造$\\#(w)$

$$
\begin{equation} \label{eq:make_w}
\begin{split}
\ell&=\sum_{w \in V_W} \sum_{c \in V_C}  \#(w,c)\log\sigma(\vec{w}\cdot\vec{c})+\sum_{w \in V_W} \sum_{c \in V_C}  \#(w,c) (k \mathbb{E}_{c_N \sim P_n(c)}[\log\sigma(-\vec{c_N}\cdot\vec{w})])\\
&=\sum_{w \in V_W} \sum_{c \in V_C}  \#(w,c)\log\sigma(\vec{w}\cdot\vec{c})+\sum_{w \in V_W}  \#(w) (k \mathbb{E}_{c_N \sim P_n(c)}[\log\sigma(-\vec{c_N}\cdot\vec{w})])
\end{split}
\end{equation}
$$

单独展开期望那一项，构造$\\#(c)$

$$
\begin{equation} \label{eq:expand_e}
\begin{split}
&\mathbb{E}_{c_N \sim P_n(c)}[\log\sigma(-\vec{c_N}\cdot\vec{w})] \\
=&\sum_{c_N \in V_C} \frac{ \#(c_N)}{ \vert D \vert } \log\sigma(-\vec{c_N}\cdot\vec{w}) \\
=&\frac{ \#(c)}{ \vert D \vert } \log\sigma(-\vec{w}\cdot\vec{c}) + \sum_{c_N \in V_C \setminus \{c\}} \frac{ \#(c_N)}{ \vert D \vert } \log\sigma(-\vec{c_N}\cdot\vec{w})
\end{split}
\end{equation}
$$

合并上面两个公式，得到具体的$(w,c)$的局部目标

\begin{equation} \label{eq:local_obj}
\ell(w,c)=\\#(w,c)\log\sigma(\vec{w}\cdot\vec{c}) + k \cdot \\#(w) \frac{\\#(c)}{ \vert D \vert } \log\sigma(-\vec{w}\cdot\vec{c})
\end{equation}

令$x=\vec{w}\cdot\vec{c}$，则对$x$求偏导

\begin{equation} \label{eq:dx}
\frac{\partial \ell}{\partial x}=\\#(w,c) \cdot \log\sigma(-x) - k \cdot \\#(w) \frac{\\#(c)}{ \vert D \vert } \log\sigma(x)
\end{equation}

令偏导数为0，并且做一些简化

\begin{equation} \label{eq:dx_0}
e^{2x}-(\frac{\\#(w,c)}{k \cdot \\#(w) \cdot \frac{\\#(c)}{ \vert D \vert }}-1)e^x-\frac{\\#(w,c)}{k \cdot \\#(w) \cdot \frac{\\#(c)}{ \vert D \vert }}=0
\end{equation}

如果令$y=e^x$，$t=\frac{\\#(w,c)}{k \cdot \\#(w) \cdot \frac{\\#(c)}{ \vert D \vert }}$，那么上面公式其实就是个简单的一元二次方程

$$
\begin{equation} \label{eq:dx_0_s}
\begin{split}
y^2-(t-1)y-t=0\\
y^2-yt+y-t=0\\
(y-t)(y+1)=0
\end{split}
\end{equation}
$$

y有两个解，一个是-1（按y的定义，舍弃），另一个解

\begin{equation} \label{eq:y_is}
y=t=\frac{\\#(w,c)}{k \cdot \\#(w) \cdot \frac{\\#(c)}{ \vert D \vert }}=\frac{\\#(w,c) \cdot  \vert D \vert }{\\#(w) \cdot \\#(c)} \cdot \frac{1}{k}
\end{equation}

按y和x的定义，可知

$$
\begin{equation} \label{eq:wc_is}
\begin{split}
\vec{w}\cdot\vec{c}
=\log \frac{ \#(w,c)}{k \cdot  \#(w) \cdot \frac{ \#(c)}{ \vert D \vert }}
&=\log\frac{ \#(w,c) \cdot  \vert D \vert }{ \#(w) \cdot  \#(c)} - \log k \\
&=\log\frac{\frac{ \#(w,c)}{ \vert D \vert }}{\frac{ \#(w)}{ \vert D \vert } \cdot \frac{ \#(c)}{ \vert D \vert }} - \log k
\end{split}
\end{equation}
$$

回顾一下PMI(Pointwise mutual information)的定义，它是用来衡量“相关性”的（a measure of association）。当$x,y$完全相关时，即$p(x \vert y)=1$或者$p(y \vert x)=1$时，PMI最大；当它们完全无关时，PMI最小，为0。

\begin{equation} \label{eq:pmi}
PMI(x;y)=\log\frac{p(x,y)}{p(x)p(y)}=\log\frac{p(x \vert y)}{p(x)}=\log\frac{p(y \vert x)}{p(y)}
\end{equation}

综合以上两个公式，那么

\begin{equation} \label{eq:wx_is}
M_{ij}^{SGNS}=W_i\cdot C_j=\vec{w_i}\cdot\vec{c_j}=PMI(w_i;c_j)-\log k
\end{equation}

到这里，我们可以得出结论：如果k=1，那么M矩阵的元素就是当前词和上下文词的PMI。word2vec中k一般取[5~20]，所以，M矩阵的元素是有偏移的PMI，即$M^{PMI_k}=M^{PMI}-\log k$。k是常数，矩阵$M$可以通过扫描语料事先计算出来，然后就可以用矩阵分解的方式来求解词向量了。

# 矩阵分解角度的目标函数 

既然word2vec跟矩阵分解有这样的等价关系，那么，我们完全可以抛弃word2vec原来的目标函数，采用矩阵分解的那一套目标函数（loss）来求解词向量，如下。矩阵分解目标函数一般长这个样子，目标是让$W_i \cdot C^\top_j$与$M_{ij}$尽量的相同，$b_i$和$b_j$分别是当前词和上下文词的bias。

$$
\begin{equation} 
\begin{split}
\mathcal{L}_{MF}=\sum_{i,j}f(M_{ij})(W_i \cdot C^{\top}_j-M_{ij}+b_i+b_j)^2
\end{split}
\label{eq:mf_loss}
\end{equation}
$$

## SGNS 

回顾下，上面小节说明了SGNS是在做矩阵分解，那么从矩阵分解角度看，它的当个pair $(w,c)$目标函数为

$$
\begin{equation} 
\begin{split}
\mathcal{L}_{SGNS}(w,c)&=f( \#(w,c))(\vec{w}\cdot\vec{c}-PMI(w;c))^2\\
&=f( \#(w,c))(\vec{w}\cdot\vec{c}-\log \#(w,c)-\log \vert D \vert +\log \#(w)+\log \#(c))^2  
\end{split}
\label{eq:sgns_mf}
\end{equation}
$$

其中用到

$$
\begin{equation} \label{eq:log_pmi}
\begin{split}
{PMI}(w;c)&=\log\frac{ \#(w,c) \cdot  \vert D \vert }{ \#(w) \cdot  \#(c)} \\
&=\log \#(w,c)+\log \vert D \vert -\log \#(w)-\log \#(c)
\end{split}
\end{equation}
$$

是不是跟(\ref{eq:mf_loss})长得很像，要不怎么说他们是等价的。另外，有研究人员直接从矩阵分解的角度求解词向量，这里介绍两个方法：GloVe和Swivel

## GloVe 

参考[ref]Pennington, GloVe: Global Vectors for Word Representation, 2014[/ref]

GloVe是<strong>Glo</strong>bal <strong>Ve</strong>ctor的简称。之所以叫Global，是因为它事先扫描了语料，统计好了全局的pair信息。它的主要目的是在推断任务（A is to B as C is to X）上beat SGNS。所以基于词向量的线性关系（推理），通过很多假设，推导出了下面这样的目标函数。

$$
\begin{equation} \label{eq:glove}
\mathcal{L}_{GloVe}(w,c)=f(\#(w,c))(\vec{w}\cdot\vec{c}-\log\#(w,c)+b_w+b_c)^2
\end{equation}
$$

GloVe目标函数的复杂度只跟pair的数量有关，跟语料的大小无关，这是它优于word2vec的地方。就目前所知，GloVe的工作并不是在Goldberg 14的基础上做的，他们是独立做出来的。但是，对比(\ref{eq:sgns_mf})(\ref{eq:glove})可以看到，他们有很多相似点：

<ol>	
    <li>都是在估计取log的共现次数</li>
    <li>GloVe有bias参数，SGNS已经给出了bias的值，即取log的词频</li>
    <li>由于$f(\#(w,c))$[footnote]$f_{SGNS}(\#(w,c))=\#(w,c)$，$f_{GloVe}=(\#(w,c)/100)^{\frac{3}{4}}$ if $\#(w,c)<100$ else $1$[/footnote]的存在，都倾向于高频词的loss大些，低频词loss小些</li>
</ol>

SGNS跟GloVe也有不同，SGNS考虑到了数据中未观测到的pair，而GloVe捕捉不到这样的信息。Swivel就是基于这点提出来的

## Swivel 

参考[ref]Noam Shazeer, Swivel: Improving Embeddings by Noticing What’s Missing, 2016[/ref]

Swivel(<strong>S</strong>ubmatrix-<strong>wi</strong>se <strong>V</strong>ector <strong>E</strong>mbedding <strong>L</strong>earner)的目标函数如下，可以看到，目标函数分两部分：对于观测到的pair，形式与SGNS和GloVe差不多；对于未观测到的pair，$\\#(w,c)$=0，这时${PMI}(w;c)=-\infty$，导致均方误差无法估计。Swivel认为未观测到不一定表示pair永远不会出现，只是这个pair比较罕见，我们的数据不够大而已。然而，不管怎样，毕竟没有见过这个pair，既不能让PMI等于$-\infty$，也不能让PMI太大。所以，假设这个pair出现了一次，利用${PMI^*}(w;c)$平滑了一下，定义了未观测到的pair所估计的PMI上界。

$$
\mathcal{L}_{Swivel}(w,c)=\left\{
\begin{array}{rcl}
f(\#(w,c))(\vec{w}\cdot\vec{c}-PMI(w;c))^2 &{\#(w,c)>0}\\
\\
\log [1+\exp(\vec{w}\cdot\vec{c}-PMI^*(w;c))] &{\#(w,c)=0}
\end{array} \right.
$$


$$
\begin{equation} \label{eq:pmi_star}
{PMI^\ast}(w;c)=\log\frac{(\#(w,c)+1) \cdot  \vert D \vert }{\#(w) \cdot \#(c)}
\end{equation}
$$

# 推断任务的实验 

这个实验是Swivel的作者做的。推断任务是这样的，A is to B as C is to X，给定ABC，要精确的计算出X。纵轴是准确率。横轴横轴是4个词的平均词频取log。可以看到，Swivel一直比SGNS好。低频词，SGNS比GloVe好；高频词，GloVe比SGNS好。

![image.png](/assets/images/swivel-glove-sgns.png)

# 工具包 

* word2vec: <a href="https://github.com/tmikolov/word2vec" target="_blank">https://github.com/tmikolov/word2vec</a>
* GloVel: <a href="https://nlp.stanford.edu/projects/glove/" target="_blank">https://nlp.stanford.edu/projects/glove/</a>
* Swivel: <a href="https://github.com/tensorflow/models/tree/master/research/swivel" target="_blank">https://github.com/tensorflow/models/tree/master/swivel</a>

# 参考文献 

* Goldberg, Neural Word Embedding as Implicit Matrix Factorization, NIPS 14 
* Pennington, GloVe: Global Vectors for Word Representation, 2014 
* Noam Shazeer, Swivel: Improving Embeddings by Noticing What’s Missing, 2016 


