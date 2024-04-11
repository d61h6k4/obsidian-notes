[Link to the paper](https://storage.googleapis.com/gweb-research2023-media/pubtools/pdf/45286.pdf)

Let $Q = (q, \{x_1,...,x_n\})$ denote a query string q and its set of result documents.
Let $P(Q)$ denote the probability of observing query $Q$, based on the underlying distribution of queries in the universe $\mathcal{Q}$ of all possible queries that users can issue together with all possible result combinations.
In practice, the empirical loss defined over a uniformly random sample $\mathcal{U} = \{Q \in \mathcal{Q} : Q \sim P(Q)\}$.
$$
L_{\mathcal{U}}(f) = \frac{1}{\mathcal{U}} \sum_{Q \in \mathcal{U}} l(Q,f)
$$
> Generally, the state-of-the-art loss functions are pair-wise or list-wise. Practically, pair-wise loss functions are more efficient for training and have been widely adopted.

Example of the pairwise loss function:
Let $x_i \succ_{Q} x_j$ denote all pairs $x_i$, $x_j$ of result documents in $Q$ for which $x_i$ is more relevant than $x_j$.
$$
l(Q,f) = \sum_{x_i \succ_{Q} x_j} max(0, f(x_j) - f(x_i))^2
$$
The intuition behind this loss function is to penalise the out-of-order pairs when ranked by $f$.

### Selection Bias Problem
The dataset $\mathcal{U}$ is the training data used to learn the scoring function $f(x)$. There are two commonly used approaches to obtain relevance estimates for $\mathcal{U}$ . (**How to get labelling)
1. Ask humans (*explicit*) - expensive and biased on the human being's opinion
2. Click-through rate (implicit) - more clicks get the document, -> more relevant it is to the query.
> **However, click data is biased and very noisy.** For example, because of *position bias*, simple click counts can not be used directly to estimate relevance. (The less relevant document in the first position gets more clicks than the more relevant document from the second because users tend to trust the system, are too lazy to go through everything, only sometimes know what they need, etc.)

**Observation 1** *When using click-through data for learning-to-rank, queries without clicks provide no useful information when optimising pair-wise loss functions.*

For example: When there are no clicks for query $Q$, the set $x_i \succ_{Q} x_j$ is empty since there is no way to derive preferences between any pairs of documents. 
In the following, we focus on the collection of queries with clicks and use $\mathcal{S}$ to demote this collection.

**Observation 2** *The collection of queries $\mathcal{S}$ is biased. Formally, let $\hat{P}(Q)$ denote the probability mass of query $Q$ in $\mathcal{S}$, then $\hat{P}(Q) \neq P(Q)$* 
An example to better explain this observation. We have two queries $Q_1$ and $Q_2$ that both have equal probability of being issued by users, i.e. $P(Q_1) = P(Q_2)$ as they have equal probability in $\mathcal{U}$. The relevant document for $Q_1$ is at position 1 and is clicked every time the query is issued. On the contrary, the relevant document for $Q_2$ is at position 2 and is clicked half of the time when the query is issued. Thus, $\hat{P}(Q_2) = \frac{1}{2}\hat{P}(Q)$ in $\mathcal{S}$, which helps illustrate how selection bias may arise in click data. The problem illustrated in this example is rooted at the commonly known *position bias* and confirmed by eye tracking studies as well, which found that the users are less likely to see, and hence click on, lower-ranked documents.

Here, we say that $Q_2$ has less chance of being selected for the training dataset if we use pure CTR. **Selection bias**.

### Inverse Propensity Weighting
With inverse propensity weighting, $\hat{P}(Q)$ is known as the *propensity score* of $Q$. Let $w_{Q} = \frac{P(Q)}{\hat{P}(Q)}$ , i.e. the ratio between the probability of $Q$ appearing in $\mathcal{U}$ and the probability that $Q$ actually appears in $\mathcal{S}$. Then, the empirical loss function becomes:
$$
L_{\mathcal{S}}(f) = \frac{1}{|\mathcal{S}|} \sum_{Q \in \mathcal{S}} \frac{P(Q)}{\hat{P}(Q)}l(Q,f) = \frac{1}{|\mathcal{S}|} \sum_{Q \in \mathcal{S}} w_Q \cdot l(Q,f)
$$

To apply selection bias in practice, the primary challenge becomes estimating the inverse propensity weights $w_Q$.

### Global bias model
In order to quantify the position bias, which will be used for inverse propensity weighting estimation, we employ result randomisation and collect user click data on the randomised result sets. Specifically, given a ranked result list of $n$ documents returned for some query, instead of showing the original list, we permute the results uniformly at random and present the shuffled list to a small fraction of end users.

In a randomised scenario the number of clicks on the $i$-th position document shows the position bias for position $i$, because only the reason to click to the document in the $i$-th position is it's position (it could be that document happened to be relevant, but experiment needs to be big enough to eliminate this case).

When we have a randomised experiment, we get position bias estimation, so $w_Q = \frac{P(Q)}{\hat{P}(Q)} \propto \frac{1}{b_i}$ 

### [Mitigating position bias](https://eugeneyan.com/writing/position-bias/#mitigating-position-bias)

If we’re in the early days of building our recommender system or prioritize exploration over exploitation, adding some randomness can be a decent way to mitigate position bias while collecting click data. Because multiple items can appear in the same position (e.g., position 1), we can log which item performed better and train our models accordingly.

If adding randomness is not an option, we can use the measured/learned position bias to debias logged data. For example, the previous Google paper used inferred position bias to train models optimized on [inverse propensity weighted precision](https://dl.acm.org/doi/10.1145/3159652.3159732).

Alternatively, we can account for position bias by including positional features in our models. These positional features help the model learn how position affects reward. Then, during serving, we can set all items to have positional feature = 1 to negate the impact of position. More in Google’s [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml#rule_36_avoid_feedback_loops_with_positional_features).