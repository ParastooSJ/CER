
import json
import sys
if len(sys.argv)==3:
  inputdata=sys.argv[1]
  input_type=sys.argv[2]

with open(inputdata, 'r') as f:
  line=json.load(f)

  file_0_test=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_0/test.json",'w')
  data_0_test=line['0']['testing']

  file_0_train=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_0/train.json",'w')
  data_0_train=line['0']['training']

  file_1_test=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_1/test.json",'w')
  data_1_test=line['1']['testing']

  file_1_train=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_1/train.json",'w')
  data_1_train=line['1']['training']


  file_2_test=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_2/test.json",'w')
  data_2_test=line['2']['testing']

  file_2_train=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_2/train.json",'w')
  data_2_train=line['2']['training']

  file_3_test=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_3/test.json",'w')
  data_3_test=line['3']['testing']

  file_3_train=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_3/train.json",'w')
  data_3_train=line['3']['training']

  file_4_test=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_4/test.json",'w')
  data_4_test=line['4']['testing']

  file_4_train=open("./data/"+str(input_type)+"/"+str(input_type)+"_fold_4/train.json",'w')
  data_4_train=line['4']['training']

inputdata="./final.json"
with open(inputdata, 'r') as f:
  for line in f:
        line=json.loads(line)

        if line["index"]in data_0_test:

            json.dump(line,file_0_test)
            file_0_test.write('\n')

        if line["index"]in data_0_train:
            
            json.dump(line,file_0_train)
            file_0_train.write('\n')


        if line["index"]in data_1_test:
            
            json.dump(line,file_1_test)
            file_1_test.write('\n')

        if line["index"]in data_1_train:
            
            json.dump(line,file_1_train)
            file_1_train.write('\n')


        if line["index"]in data_2_test:
            
            json.dump(line,file_2_test)
            file_2_test.write('\n')

        if line["index"]in data_2_train:
            
            json.dump(line,file_2_train)
            file_2_train.write('\n')

        if line["index"]in data_3_test:
            
            json.dump(line,file_3_test)
            file_3_test.write('\n')

        if line["index"]in data_3_train:
            
            json.dump(line,file_3_train)
            file_3_train.write('\n')

        if line["index"]in data_4_test:
            
            json.dump(line,file_4_test)
            file_4_test.write('\n')

        if line["index"]in data_4_train:
            
            json.dump(line,file_4_train)
            file_4_train.write('\n')


file_0_test.close()
file_0_train.close()

file_1_test.close()
file_1_train.close()

file_2_test.close()
file_2_train.close()

file_3_test.close()
file_3_train.close()

file_4_test.close()
file_4_train.close()

