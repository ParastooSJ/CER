cd code
python random-sample-generator.py all 0 train
python sample-scorer.py all 0 train
python data-constructor.py all 0 train
python data-constructor.py all 0 test
python subgraph-pruning-trainer.py all 0
python subgraph-pruner.py all 0
python subgraph-truncate.py all 0
python subgraph-ranker.py all 0

python random-sample-generator.py all 1 train
python sample-scorer.py all 1 train
python data-constructor.py all 1 train
python data-constructor.py all 1 test
python subgraph-pruning-trainer.py all 1
python subgraph-pruner.py all 1
python subgraph-truncate.py all 1
python subgraph-ranker.py all 1

python random-sample-generator.py all 2 train
python sample-scorer.py all 2 train
python data-constructor.py all 2 train
python data-constructor.py all 2 test
python subgraph-pruning-trainer.py all 2
python subgraph-pruner.py all 2
python subgraph-truncate.py all 2
python subgraph-ranker.py all 2

python random-sample-generator.py all 3 train
python sample-scorer.py all 3 train
python data-constructor.py all 3 train
python data-constructor.py all 3 test
python subgraph-pruning-trainer.py all 3
python subgraph-pruner.py all 3
python subgraph-truncate.py all 3
python subgraph-ranker.py all 3

python random-sample-generator.py all 4 train
python sample-scorer.py all 4 train
python data-constructor.py all 4 train
python data-constructor.py all 4 test
python subgraph-pruning-trainer.py all 4
python subgraph-pruner.py all 4
python subgraph-truncate.py all 4
python subgraph-ranker.py all 4

