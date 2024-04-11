[Link to the paper](https://arxiv.org/pdf/1810.09591.pdf)
The paper tells the history of applying NN for a search in AirBnB.
The primary offline metric is NDCG.

##### Loss function
Cross entropy: $H(p,q) = -\sum_{x \in X}p(x)\log(q(x))$
Cross entropy loss function:
$p \in \{y, 1 - y\}$ - ground truth and $q \in \{\hat{y}, 1-\hat{y}\}$ - predictions
$H(p,q) = -y\log(\hat{y}) - (1 - y)\log(1 - \hat{y})$
Loss is weird, btw, or most probably I didn't get it.

### Feature normalisation
Feeding values outside the usual range of features can cause large gradients to backpropagate. This can permanently shut off activation functions like ReLU due to vanishing gradients. To avoid it, we ensure all features are restricted to a small range of values, with the bulk of the distribution in the {-1, 1} interval and the median mapped to 0. This, by and large, involves inspecting the features and applying either of the two transforms:
> * In case the feature distribution resembles a normal $\frac{feature_{val} - \mu}{\sigma}$
> * If the feature distribution looks close to a power law distribution, we transform it by $\log(\frac{1+feature_{val}}{1 + median})$

> Why obsess over the smoothness of distributions?
> **Spotting bugs** 
> **Facilitating generalisation - preserving distribution of the feature from layer to layer
> Checking feature completeness. In some cases, investigat- ing the lack of smoothness of certain features lead to the discovery of features the model was missing.



