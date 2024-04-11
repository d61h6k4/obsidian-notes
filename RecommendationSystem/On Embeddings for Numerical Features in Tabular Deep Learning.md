[Link](https://arxiv.org/pdf/2203.05556.pdf)
embeddings for numerical features: the first one is based on a piecewise linear encoding of scalar values, and the second one utilizes periodic activations.
**CTR Prediction**. In CTR prediction problems, objects are represented by numerical and categorical features, which makes this field highly relevant to tabular data problems. In several works, numerical features are handled in some non-trivial way while not being the central part of the research [8, 40]. Recently, however, a more advanced scheme has been proposed in Guo et al. [14]. Nevertheless, it is still based on linear layers and conventional activation functions, which we found to be suboptimal in our evaluation.

### PLE
First step: create bins per feature. The number of bins per feature is controled with variable $T$.
PLE (stands for peicewise linear encoding):
$PLE(x) = [e_1,...,e_T] \in \mathbb{R}^T$
$$
\begin{cases} 
      0 & x\le b_{t-1} \text{ and } t > 1 \\
      1 & x \geq b_t \text{ and } t < T \\
      \frac{x - b_{t-1}}{b_t - b_{t-1}} & \text{otherwise} 
   \end{cases}
$$
##### A note on attention-based models.
attention-based models are inherently invariant to the order of input embeddings, so one additional step is required to add the information about feature indices to the obtained encodings. So the idea is to use learning positional encoding, which means:
$f_i(x) = v_0 + \sum_{t=1}^T e_t \cdot v_t = \text{Linear(PLE(x))}$

##### How to obtain bins?
1. quantiles: $b_t = Q_{\frac{t}{T}}(\{x_i^{j(num)}\}_{j \in J_{train}})$
2. supervised approach, similar to C4.5 discretisation algorithm. In a nutshell, for each feature, we recursively greedily split its value range using the target as guidance, which is equivalent to building a decision tree (which is used for growing only this one feature and the target) and treating the regions corresponding to its leaves as the bins for PLE . Additionally, we define $b^i_0 = \min_{j \in J_train} x^j_i$ and $b^i_T = \max_{j \in J_{train}} x^j_i$ .


 ### Periodic activation functions
 $$
 f_i(x) = \text{Period}(x) = concat[\sin(v),\cos(v)], \quad v = [2\pi c_1x, ... , 2\pi c_k x]
 $$
 where $c_i$ are trainable parameters initialised from $\mathcal{N}(0,\sigma)$ . They observe that $\sigma$ is an important hyperparameter. Both $\sigma$ and $k$ are tuned using validation sets.

 ## Implementation details
 
 They also apply standardisation to regression targets for all algorithms. 