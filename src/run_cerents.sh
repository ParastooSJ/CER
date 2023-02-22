python main.py all 0 CERENTS 
python main.py all 1 CERENTS
python main.py all 2 CERENTS
python main.py all 3 CERENTS
python main.py all 4 CERENTS
python main.py INEX 0 CERENTS
python main.py INEX 1 CERENTS
python main.py INEX 2 CERENTS
python main.py INEX 3 CERENTS
python main.py INEX 4 CERENTS
python main.py QALD2 0 CERENTS
python main.py QALD2 1 CERENTS
python main.py QALD2 2 CERENTS
python main.py QALD2 3 CERENTS
python main.py QALD2 4 CERENTS
python main.py ListSearch 0 CERENTS
python main.py ListSearch 1 CERENTS
python main.py ListSearch 2 CERENTS
python main.py ListSearch 3 CERENTS
python main.py ListSearch 4 CERENTS
python main.py SemSearch 0 CERENTS
python main.py SemSearch 1 CERENTS
python main.py SemSearch 2 CERENTS
python main.py SemSearch 3 CERENTS
python main.py SemSearch 4 CERENTS

cat ../data/all/all_fold_0/CERENTS_final.txt ../data/all/all_fold_1/CERENTS_final.txt ../data/all/all_fold_2/CERENTS_final.txt ../data/all/all_fold_3/CERENTS_final.txt ../data/all/all_fold_4/CERENTS_final.txt >../Results/CER_ENTS_All.txt
cat ../data/INEX/INEX_fold_0/CERENTS_final.txt ../data/INEX/INEX_fold_1/CERENTS_final.txt ../data/INEX/INEX_fold_2/CERENTS_final.txt ../data/INEX/INEX_fold_3/CERENTS_final.txt ../data/INEX/INEX_fold_4/CERENTS_final.txt  > ../Results/CER_ENTS_INEX.txt
cat ../data/QALD2/QALD2_fold_0/CERENTS_final.txt ../data/QALD2/QALD2_fold_1/CERENTS_final.txt ../data/QALD2/QALD2_fold_2/CERENTS_final.txt ../data/QALD2/QALD2_fold_3/CERENTS_final.txt ../data/QALD2/QALD2_fold_4/CERENTS_final.txt > ../Results/CER_ENTS_QALD2.txt
cat ../data/ListSearch/ListSearch_fold_0/CERENTS_final.txt ../data/ListSearch/ListSearch_fold_1/CERENTS_final.txt ../data/ListSearch/ListSearch_fold_2/CERENTS_final.txt ../data/ListSearch/ListSearch_fold_3/CERENTS_final.txt ../data/ListSearch/ListSearch_fold_4/CERENTS_final.txt > ../Results/CER_ENTS_ListSearch.txt
cat ../data/SemSearch/SemSearch_fold_0/CERENTS_final.txt ../data/SemSearch/SemSearch_fold_1/CERENTS_final.txt ../data/SemSearch/SemSearch_fold_2/CERENTS_final.txt ../data/SemSearch/SemSearch_fold_3/CERENTS_final.txt ../data/SemSearch/SemSearch_fold_4/CERENTS_final.txt > ../Results/CER_ENTS_SemSearch.txt