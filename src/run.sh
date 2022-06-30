cd src
python random-sample-generator.py all 0 train
python sample-scorer.py all 0 train
python data-constructor.py all 0 train
python data-constructor.py all 0 test
python subgraph-pruning-trainer.py all 0
python subgraph-pruner.py all 0
python subgraph-truncate.py all 0
python subgraph-ranker.py all 0
