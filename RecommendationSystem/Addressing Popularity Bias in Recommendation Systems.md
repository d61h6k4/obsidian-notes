#### How do we measure popularity bias?
##### Average Recommendation Popularity (ARP)
The metric is used to evaluate the popularity of recommended items in a list. It calculates the average popularity of the items based on the number of ratings they have received in the training set. 
$$
ARP = \frac{1}{|U_t|} \sum_{u \in U_t} \frac{1}{|L_u|} \sum_{i \in L_u}\phi(i)
$$
Where $U_t$ is the number of users, $L_u$ is the number of items in the recommended list for the user $u$. $\phi(i)$ is the number of times the item $i$ has been rated in the training set.
In simple terms, ARP measures the average popularity of items in the recommended lists by summing up the popularity (number of ratings) of all items in those lists and then averaging this popularity across all users in the test set.

#### The Average Percentage of Long Tail items (APLT)
The metric calculates the average proportion of long-tail items present in recommended lists. 
$$
APLT = \frac{1}{|U_t|} \sum_{u \in U_t} \frac{|L_u \cap \Gamma|}{|L_u|}
$$
Where $\Gamma$ is the set of long-tail items.

#### How do we reduce popularity bias?
This idea takes inspiration from position-aware learning (PAL), where the approach to rank suggests asking your ML model to optimise both ranking relevancy and position impact simultaneously. We can use the same approach with a popularity score; this score can be any of the above scores, like Average Recommendation Popularity.

- On training time, you use item popularity as one of the input features.
- In the prediction stage, you replace it with a constant value.

> I would suggest taking the popularity itself and scaling it with distribution (Search in AirBnB paper)

