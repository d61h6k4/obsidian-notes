[Link](https://arxiv.org/pdf/2403.00793.pdf)
several fundamental questions regarding representation learning in large-scale real-world ad recommenders remain unanswered:
* *Priors for Representation*: Real-world systems encompass various types of features from diverse sources, including sequence features (e.g., user click/conversion history), numeric features (e.g., semantic-preserving ad IDs), and embedding features from pre-trained external models. Preserving the inherent priors of these features while encoding them in recommendation systems is crucial.
* *DimensionalCollapse*: The encoding process maps all features into embeddings, typically represented as 𝐾-dimensional vectors, which are learned during model training. However, we observe that the embeddings of many fields tend to occupy a lower-dimensional subspace instead of fully utilising the available 𝐾-dimensional space. Such dimensional collapse leads to parameter wastage and limits the scalability of recommendation models.
* *Interest Entanglement*: User responses in ad recommender systems are influenced by complex interest among vari- ous factors, particularly when multiple tasks or scenarios are learned simultaneously. Existing shared-embedding approaches fail to disentangle these factors adequately, as they rely on a single entangled embedding for each feature.

#### Feature Encoding

##### Sequence Features
A user’s history behaviors reflect her interest, making them critical in recommendations. One key characteristic of such features is that there are strong semantic as well as temporal correlations between these behaviors and the target. For example, given a target, those behaviors that are either semantically related (e.g., belonging to the same category with target ad) or temporally close are more informative to predict the user’s response.
We propose Temporal Interest Module (TIM) to learn the above-mentioned four-way semantic-temporal correlation between quadruplet (behavior semantic, target semantic, behavior temporal, target temporal)

##### Numerical Features
Different from the widely-used categorical features, there is inherent partial order between numeric/ordinal features, such as Age_20 ≺ Age_30. To preserve these partial priors, inspired by the n-ary encoding, we propose a simple yet efficient variant, namely Multiple Numeral System (MNS). Formally,
$$
f_{MNS}(\cdot) = \sum_{k = 1}^{K_2}X_{2k+\mathbb{B}_k} + \sum_{k=1}^{K_3}X_{3k+\mathbb{C}_k} + ... + \sum_{k=1}^{K_n} X_{nk + \mathbb{N}_k}
$$
where $\mathbb{B} =$ func_binary$(v)$ , $\mathbb{C}=$ func_ternary$(v)$,... and $K_2$, $K_3$ are the lengths of the encoding list for binary and ternary systems, respectively; func_binary and func_ternary are the binarisation and ternarization functions, respectively.

**It's an interesting idea how to encode ad IDs, so we have a distance function between them.**
In an advertising system, ads are often indexed by discrete identifiers (Ad IDs), which are self-incremental IDs and, hence, meaningless. However, each ad is associated with a creative containing abundant visual semantics. Consequently, we replace the Ad IDs by a novel HashID to preserve the visual semantics. Specifically, we first get visual embeddings of ads from a vision model based on their creatives, then apply hashing algorithms such as Locality-Sensitive Hashing (LSH) on them to preserve the visual distances, and finally attain the HashIDs. In this way, ads with similar appearance have contiguous HashIDs, i.e., the hamming distance between the hash coding lists of two similar ads will be smaller than that of two dissimilar ads. Therefore, HashIDs can be regarded as special numeric features, and we further apply MNS to them to preserve their ordinal priors.

##### Embedding Features
Besides the main recommendation model, we may train a separate model, such as LLM or GNN, to learn embeddings for entities (users or items). Such embeddings capture the relationship between users and items from a different perspective, e.g., as a Graph or Self-Supervised Language Model, and hence should provide extra information to the recommendation models. The key challenge in leveraging such pre-trained embedding directly in our recommendation system is the semantic gap. That is, these embedding captures different semantics from the collaborative semantics of the ID embeddings in recommendation models [33, 69]. For example, LLM embeddings learn a language semantic, whilst GNN embeddings learn a graph semantic, using a cosine distance [22] rather than inner product distance in matrix factorization-based recommenders.
**How do they use embedding features?**
1. They calculate the similarity (in terms of the initial model, e.g. LLM or GNN) of objects (e.g. user and item) $w_{sim} = sim(v_u, v_i)$ 
2. $e_{sim} = f_{MNS}(w_{sim})$ - encode the ordered numerical feature (see above)
3. use user_id, item_id and $e_{sim}$ as features.

We employ self-supervised pre-training using GraphSage on a user-ad/content bipartite graph, with clicks in both ad and content recommendation domains as the edges.

#### Tackling dimensional collapse
**Multi-Embedding Paradigm** - Specifically, we scale up the number of embedding tables instead of the embedding size and incorporate embedding-table-specific feature interaction modules. Given a feature, we look up several embeddings for it, each from a different embedding table. **Then all feature embeddings from the same embedding table interact with each other in the corresponding feature interaction module.**
**One requirement of multi-embedding** is that there should be non-linearities such as ReLU after feature interaction; otherwise, the model is equivalent to single-embedding and hence does not capture different patterns.

#### Model Training
* Gradient Vanishing and Ranking Loss (we covered it [here](obsidian://open?vault=Obsidian%20Vault&file=Understanding%20the%20Ranking%20Loss%20for%20Recommendation%20with%20Sparse%20User%20Feedback))
* Repeated Exposure and Weighted Sampling - give special weight to the loss for repeated ads, introduces bias. Read the 6.2 chapter to learn how to mitigate the bias.
* Online learning (fine-tuning the model on the fresh data)
* Exploration with Uncertainty Estimates - use ThompsonSampling for exploration-exploitation trade-off