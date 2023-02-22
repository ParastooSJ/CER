python main.py all 0 CERDIS 
python main.py all 1 CERDIS
python main.py all 2 CERDIS
python main.py all 3 CERDIS
python main.py all 4 CERDIS
python main.py INEX 0 CERDIS
python main.py INEX 1 CERDIS
python main.py INEX 2 CERDIS
python main.py INEX 3 CERDIS
python main.py INEX 4 CERDIS
python main.py QALD2 0 CERDIS
python main.py QALD2 1 CERDIS
python main.py QALD2 2 CERDIS
python main.py QALD2 3 CERDIS
python main.py QALD2 4 CERDIS
python main.py ListSearch 0 CERDIS
python main.py ListSearch 1 CERDIS
python main.py ListSearch 2 CERDIS
python main.py ListSearch 3 CERDIS
python main.py ListSearch 4 CERDIS
python main.py SemSearch 0 CERDIS
python main.py SemSearch 1 CERDIS
python main.py SemSearch 2 CERDIS
python main.py SemSearch 3 CERDIS
python main.py SemSearch 4 CERDIS

cat ../data/all/all_fold_0/CERDIS_final.txt ../data/all/all_fold_1/CERDIS_final.txt ../data/all/all_fold_2/CERDIS_final.txt ../data/all/all_fold_3/CERDIS_final.txt ../data/all/all_fold_4/CERDIS_final.txt >../Results/CER_DIS_All.txt
cat ../data/INEX/INEX_fold_0/CERDIS_final.txt ../data/INEX/INEX_fold_1/CERDIS_final.txt ../data/INEX/INEX_fold_2/CERDIS_final.txt ../data/INEX/INEX_fold_3/CERDIS_final.txt ../data/INEX/INEX_fold_4/CERDIS_final.txt  > ../Results/CER_DIS_INEX.txt
cat ../data/QALD2/QALD2_fold_0/CERDIS_final.txt ../data/QALD2/QALD2_fold_1/CERDIS_final.txt ../data/QALD2/QALD2_fold_2/CERDIS_final.txt ../data/QALD2/QALD2_fold_3/CERDIS_final.txt ../data/QALD2/QALD2_fold_4/CERDIS_final.txt > ../Results/CER_DIS_QALD2.txt
cat ../data/ListSearch/ListSearch_fold_0/CERDIS_final.txt ../data/ListSearch/ListSearch_fold_1/CERDIS_final.txt ../data/ListSearch/ListSearch_fold_2/CERDIS_final.txt ../data/ListSearch/ListSearch_fold_3/CERDIS_final.txt ../data/ListSearch/ListSearch_fold_4/CERDIS_final.txt > ../Results/CER_DIS_ListSearch.txt
cat ../data/SemSearch/SemSearch_fold_0/CERDIS_final.txt ../data/SemSearch/SemSearch_fold_1/CERDIS_final.txt ../data/SemSearch/SemSearch_fold_2/CERDIS_final.txt ../data/SemSearch/SemSearch_fold_3/CERDIS_final.txt ../data/SemSearch/SemSearch_fold_4/CERDIS_final.txt > ../Results/CER_DIS_SemSearch.txt