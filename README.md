# CER
#### **CER** is a Contextual Entity Retrieval method that models contextual relationship between entities and effectively limits the extensive search space without compromising performance. In this method, a model is trained to prune an extracted subgraph from a textual knowledge graph that represents the relations between entities and then a second deep model is trained to rank entities in the subgraph by reasoning over the textual content of nodes, edges, and the given query. We added two different variations of CER called CERDIS that uses entities description instead of entities titles and CERENTS that uses the question's entities instead of the question itself.


## Code and DATA
#### we have provided our code in src folder that can be used for producing the results. The extracted data from contextual Wikipedia knowledge graph for DBPedia-Entity v2 can be found in data folder. We have put our runs in Results folder.

#### To run the code, follow the instruction below.
1) clone the git repository.
```
git clone https://github.com/ParastooSJ/CER.git
cd CER
```
2) Install all the required packages in requirements.txt.
3) To run the code using the pre-trained models, download the models from [here](https://drive.google.com/drive/folders/1e12XvXv7gaUvaCEC33kEhbQBVLjzYsuk?usp=share_link) and place them in their respective subfolders in model folder, otherwise it would train the models from the scratch.
4) Run the following code to generate the results.
```
cd src\
bash run.sh
```

## Citation
#### If you find our code and data useful, please cite our paper.
```
@inproceedings{jafarzadeh2022learning,
  title={Learning to Rank Knowledge Subgraph Nodes for Entity Retrieval},
  author={Jafarzadeh, Parastoo and Amirmahani, Zahra and Ensan, Faezeh},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={2519--2523},
  year={2022}
}
```
