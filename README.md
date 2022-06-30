# CER
#### **CER** is a Contextual Entity Retrieval method that models contextual relationship between entities and effectively limits the extensive search space without compromising performance. In this method, a model is trained to prune an extracted subgraph from a textual knowledge graph that represents the relations between entities and then a second deep model is trained to rank entities in the subgraph by reasoning over the textual content of nodes, edges, and the given query.


## Code and Results
#### we have provided our code and results in src and runs sections that can be used for replicating the results. The extracted data from contextual Wikipedia knowledge graph for DBPedia-Entity v2 can be found here.

#### To run the code, place the downloaded file in data folder and run the following commands:
```
cd src\data\
bash split.sh
cd ..
bash run.sh

```
