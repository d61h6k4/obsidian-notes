[Link to the paper](https://daiwk.github.io/assets/youtube-multitask.pdf)

Typical recommendation systems follow a two-stage design with a candidate generation and a ranking. This paper focuses on the ranking stage. In this stage, the recommender has a few hundred candidates retrieved from the candidate generation (e.g. two tower model). It applies a sophisticated large-capacity model to rank and sort the most promising items.
**We present experiments and lessons learned from building such a ranking system on a large-scale industrial video publishing and sharing platform.**
Models used here are Wide'n'Deep and MMoE. *In addition, it introduces a shallow tower to model and remove selection bias.* - This is the rule #36 of [Rules of Machine Learning](obsidian://open?vault=Obsidian%20Vault&file=Rules%20of%20Machine%20Learning)

We first group our multiple objectives into two categories: 1) engagement objectives, such as user clicks, and degree of engagement with recommended videos; 2) satisfaction objectives, such as user liking a video on YouTube, and leaving a rating on the recommendation.

![[RecWhatVideoToWatchArchitecture.png]]
See how the bias features are treated.

#### Candidate Generation
Our video recommendation system uses multiple candidate generation algorithms, each of which captures one aspect of similarity between query video and candidate video. For example, one algorithm generates candidates by matching topics of query video. Another algorithm retrieves candidate videos based on how often the video has been watched together with the query video. We construct a sequence model for generating personalized candidate given user history. We also use techniques to generate context-aware high recall relevant candidates.

#### System overview
Our ranking system learns from two types of user feedback: 1) engagement behaviors, such as clicks and watches; 2) satisfaction behaviors, such as likes and dismissals.

Given a query, candidate, and context, the ranking model predicts the probabilities of user taking actions such as clicks, watches, likes, and dismissals. This approach of making predictions for each candidate is a point-wise approach [6]. In contrast, pair-wise or list-wise approaches learn to make predictions on ordering of two or multiple candidates. Pair-wise or list-wise approaches can be used to potentially improve the diversity of the recommendations. **However, we opt to use point-wise ranking mainly based on serving considerations. At serving time, point-wise ranking is simple and efcient to scale to a large number of candidates. In comparison, pair-wise or list-wise approaches need to score pairs or lists multiple times in order to fnd the optimal ranked list given a set of candidates, thereby limiting their scalability.**
Engagement objectives capture user behaviors such as clicks and watches. We formulate the prediction of these behaviors into two types of tasks: *binary classifcation task for behaviors such as clicks*, and *regression task for behaviors related to time spent*.
For example, behavior such as clicking like for a video is formulated as a binary classifcation task, and behavior such as rating is formulated as regression task. 

> For binary classifcation tasks, we compute cross entropy loss.
> And for regression tasks, we compute squared loss.

For each candidate, we take the input of these multiple predictions, and output a combined score using a combination function in the form of weighted multiplication. **The weights are manually tuned to achieve best performance** on both user engagements and user satisfactions.

#### Modeling and Removing Position and Selection Biases
> Authors mentioned that as input for "shallow" tower they use not only "bias" features, but e.g. device info

Here is the explanation:
In training, the positions of all impressions are used, with a 10% feature drop-out rate to prevent
our model from over-relying on the position feature. At serving time, position feature is treated as missing. The reason why we cross position feature with device feature is that diferent position
biases are observed on diferent types of devices.

#### Experiments
For ofine experiments, we monitor AUC for classifcation task and squared error for regression tasks. For live experiments, we conduct A/B testing comparing with production system.

We examine multiple engagement metrics such as time spent at YouTube, and satisfaction metrics such as rate of dismissals, user survey responses, etc. In addition to live metrics, we also care about the computation cost of the model at serving time, since YouTube responds a substantially large number of queries per second.

###### Gating Network Stability. 
When training neutral network models using multiple machines, distributed training strategies can cause models to diverge frequently. An example of divergences is Relu death [1]. In MMoE, the softmax gating networks have been reported [32] to have imbalanced expert distribution problem, where gating networks converge to have most zero-utilization on experts. With distributed training, we observe 20% chance of this gating network polarization issue in our models. Gating network polarization harms model performance on tasks using polarized gating networks. To solve this problem, we apply drop-out on the gating networks. By applying a 10% probability of setting utilization of experts to 0 and re-normalizing the softmax outputs, we eliminate the gating network polarization for all gating networks.

