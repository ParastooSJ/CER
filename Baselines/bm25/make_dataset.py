import json


in_f = open('/home/jparastoo/downloads/dbpedia-v2/bm25/fsdm-raw.txt','r')
help_ff = open('/home/jparastoo/downloads/dbpedia-v2/bm25/bm25-data.json','r')
out_file = open('/home/jparastoo/downloads/dbpedia-v2/bm25/fsdm_made_data.json','w')
rank_f = open('/home/jparastoo/downloads/dbpedia-v2/new_results/qrels-v2.txt','r')
in_dict = {}
q_rel = {}

for line in rank_f:
    splits = line.split()
    key = splits[0]
    if key not in q_rel:
        q_rel[key] = {}
    else:
        value = splits[2].replace('<dbpedia:','')
        value = value.replace('>','')
        value = value.replace('_',' ')
        q_rel[key][value] = int(splits[3])

for line in in_f:
    splits = line.split()
    key = splits[0]
    if key not in in_dict:
        in_dict[key] = {}
    else:
        value = splits[2].replace('<dbpedia:','')
        value = value.replace('>','')
        value = value.replace('_',' ')
        in_dict[key][value] = {}

for line in help_ff:
    non = 0
    jline = json.loads(line)
    if jline['index'] in in_dict.keys():
        out_line = {"index":jline['index'], "question":jline['question'], "candidates":{}}
        for key in in_dict[jline['index']]:
            if key in jline['candidates']:
                out_line['candidates'][key] = jline['candidates'][key]
            else:
                if key in q_rel[jline['index']]:
                    rank = q_rel[jline['index']][key]
                    ans = False
                    if rank>0:
                        ans =True
                    out_line['candidates'][key] = {'rank':rank, 'ans':ans}
                    non +=1
    print(non)

    out_file.write(json.dumps(out_line)+'\n')

out_file.close() 

        