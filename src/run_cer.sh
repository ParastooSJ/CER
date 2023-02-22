python main.py all 0 CERCONTEXT 
python main.py all 1 CERCONTEXT
python main.py all 2 CERCONTEXT
python main.py all 3 CERCONTEXT
python main.py all 4 CERCONTEXT
python main.py INEX 0 CERCONTEXT
python main.py INEX 1 CERCONTEXT
python main.py INEX 2 CERCONTEXT
python main.py INEX 3 CERCONTEXT
python main.py INEX 4 CERCONTEXT
python main.py QALD2 0 CERCONTEXT
python main.py QALD2 1 CERCONTEXT
python main.py QALD2 2 CERCONTEXT
python main.py QALD2 3 CERCONTEXT
python main.py QALD2 4 CERCONTEXT
python main.py ListSearch 0 CERCONTEXT
python main.py ListSearch 1 CERCONTEXT
python main.py ListSearch 2 CERCONTEXT
python main.py ListSearch 3 CERCONTEXT
python main.py ListSearch 4 CERCONTEXT
python main.py SemSearch 0 CERCONTEXT
python main.py SemSearch 1 CERCONTEXT
python main.py SemSearch 2 CERCONTEXT
python main.py SemSearch 3 CERCONTEXT
python main.py SemSearch 4 CERCONTEXT
cat ../data/all/all_fold_0/CERCONTEXT_final.txt ../data/all/all_fold_1/CERCONTEXT_final.txt ../data/all/all_fold_2/CERCONTEXT_final.txt ../data/all/all_fold_3/CERCONTEXT_final.txt ../data/all/all_fold_4/CERCONTEXT_final.txt >../Results/CER_CONTEXT_All.txt
cat ../data/INEX/INEX_fold_0/CERCONTEXT_final.txt ../data/INEX/INEX_fold_1/CERCONTEXT_final.txt ../data/INEX/INEX_fold_2/CERCONTEXT_final.txt ../data/INEX/INEX_fold_3/CERCONTEXT_final.txt ../data/INEX/INEX_fold_4/CERCONTEXT_final.txt  > ../Results/CER_CONTEXT_INEX.txt
cat ../data/QALD2/QALD2_fold_0/CERCONTEXT_final.txt ../data/QALD2/QALD2_fold_1/CERCONTEXT_final.txt ../data/QALD2/QALD2_fold_2/CERCONTEXT_final.txt ../data/QALD2/QALD2_fold_3/CERCONTEXT_final.txt ../data/QALD2/QALD2_fold_4/CERCONTEXT_final.txt > ../Results/CER_CONTEXT_QALD2.txt
cat ../data/ListSearch/ListSearch_fold_0/CERCONTEXT_final.txt ../data/ListSearch/ListSearch_fold_1/CERCONTEXT_final.txt ../data/ListSearch/ListSearch_fold_2/CERCONTEXT_final.txt ../data/ListSearch/ListSearch_fold_3/CERCONTEXT_final.txt ../data/ListSearch/ListSearch_fold_4/CERCONTEXT_final.txt > ../Results/CER_CONTEXT_ListSearch.txt
cat ../data/SemSearch/SemSearch_fold_0/CERCONTEXT_final.txt ../data/SemSearch/SemSearch_fold_1/CERCONTEXT_final.txt ../data/SemSearch/SemSearch_fold_2/CERCONTEXT_final.txt ../data/SemSearch/SemSearch_fold_3/CERCONTEXT_final.txt ../data/SemSearch/SemSearch_fold_4/CERCONTEXT_final.txt > ../Results/CER_CONTEXT_SemSearch.txt