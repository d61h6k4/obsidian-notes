[Link](https://arxiv.org/pdf/2403.14144.pdf)

Click-through rate (CTR) prediction holds significant importance in the realm of online advertising. While many existing approaches treat it as a binary classification problem and utilize binary cross entropy (BCE) as the optimization objective, recent advancements have indicated that combining BCE loss with ranking loss yields substantial performance improvements.

##### Binary Cross Entropy
$$
\mathcal{L}_{BCE} = - \frac{1}{N} \sum_{i \in N} [y_i \log(\sigma(z_i)) + (1-y_i)\log(1 - \sigma(z_i))]
$$
where $N$ denotes the number of samples, $z_i$ represents the logit of $i$-th sample, and $y_i$ denotes the corresponding binary label, 1 for click and 0 for non-click.

##### Learning to Rank
In scenarios such as contextual advertising, however, the point-wise approach often falls into sub-optimality. Firstly, the pointwise approach treats each document as an individual input object, desire- regarding the relative order between documents. Secondly, it fails to consider the query-level and position-based properties of evaluation measures for ranking.
Specifically, pairwise and listwise approaches are the two main branches of LTR. 
*Pairwise* methods aim to ensure that the estimated value of positive samples is greater than that of negative samples for each pair of positive/negative samples.
$$
\mathcal{L}_{RankNet} = - \frac{1}{N^2} \sum_{i,j \in N} [y_{ij}\log(\sigma(z_i - z_j)) + (1 - y_{ij})\log(1 - \sigma(z_i - z_j))]
$$
where $y_{ij} \in \{0, 0.5, 1\}$ , corresponding to the conditions that $y_i < y_j, y_i = y_j, y_i > y_j$ 
Following enhancements in the optimisation process [2](https://papers.nips.cc/paper_files/paper/2006/file/af44c4c56f385c43f2529f9b1b018f6a-Paper.pdf) and the incorporation of hinge loss [33](https://chbrown.github.io/kdd-2013-usb/workshops/ADKDD/doc/wks_submission_4.pdf) have led to further improvements in performance.

*Listwise* methods encourage positive samples to have high rankings within the list of all samples
$$
\mathcal{L}_{ListNet} = - \frac{1}{N} \sum_{i \in N} \log(\frac{\exp(z_i)}{\sum_{k \in N} \exp(z_k)})
$$
Recently, studies have revealed that such methods lack calibration capabilities, potentially leading to training instability. Consequently, improved approaches like CalSoftmax have been proposed

##### Combined-Pair
$$
\mathcal{L}^{CP} = \alpha \mathcal{L}_{BCE} + (1 - \alpha)\mathcal{L}_{RankNet}
$$
##### Gradient of BCE Loss for Negative Sample.
$$
\nabla_{z_j^{(-)}} \mathcal{L}_{BCE} = \frac{1}{N} \frac{1}{1 - \sigma{z_j^{(-)}}} \sigma(z_j^{(-)})(1 - \sigma(z_j^{(-)})) = \frac{1}{N} \sigma(z_j^{(-)}) = \frac{1}{N}\hat{p}_j
$$
This equation demonstrates that the gradient of the negative sample is proportional to its pCTR value, $\hat{p}$ . The expected value of $\hat{p}$ produced by an unbiased CTR estimation model with BCE loss is close to the underlying global CTR, which latter represents the proportion of click samples (i.e., positive feedback) to the total samples.
When the positive feedback is sparse, $\hat{p}$ becomes a small value. 

> When there is sparse positive feedback, the gradients of negative samples vanish since it's proportional to the estimated positive rate.

##### Gradient of Combined-Pair for Negative Sample.
> When there is sparse positive feedback, Combined-Pair has larger gradients for negative samples than BCE method.

