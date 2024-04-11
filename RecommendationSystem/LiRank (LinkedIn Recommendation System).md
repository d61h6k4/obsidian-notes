[Link to the paper](https://arxiv.org/pdf/2402.06859.pdf)
#### Feed Ranking Model
The primary Feed ranking model employs a point-wise ranking approach, predicting multiple action probabilities including like, comment, share, vote, and long dwell and click for each <mem- ber, candidate post> pair.

> LogLoss is the candidate for a point-wise ranking loss. In the paper [Learning from Negative User Feedback and ...](https://arxiv.org/pdf/2308.12256.pdf), authors use negative log-likelihood as a point-wise ranking loss.

These predictions are linearly combined to generate the final post score.
A TF model with a multi-task learning (MTL) architecture generates these probabilities in two towers: the click tower for probabilities of click and long dwell, and contribution tower for contribution and related predictions.

> We give a tower to each task, and then each tower can have several heads.

Both towers use the same set of dense features normalized based on their distribution[13], and apply multiple fully-connected layers.

> See [Applying Deep Learning To AirBnB search](obsidian://open?vault=Obsidian%20Vault&file=Applying%20Deep%20Learning%20To%20Airbnb%20Search)

Sparse ID embedding features (§A.1) are transformed into dense embeddings [22] through lookup in embedding tables of Member/Actor and Hashtag Embedding Table

> A.1 contains an explanation of which IDs specifically were used (e.g. actor, historical actor IDs, which are creators who frequently interacted in the past with the creator of the post)
> Interesting technical detail: We empirically found 30 dimensions optimal for the ID embeddings. The sparse ID embedding features mentioned above are concatenated with all other dense features and passed through a multi-layer perception (MLP) of 4 connected layers, each with an output dimension of 100.

