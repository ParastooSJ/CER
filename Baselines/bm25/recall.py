import json



in_f = open('/home/jparastoo/downloads/dbpedia-v2/bm25/fsdm-raw.txt','r')
help_ff = open('/home/jparastoo/downloads/dbpedia-v2/bm25/bm25-data.json','r')

rank_f = open('/home/jparastoo/downloads/dbpedia-v2/new_results/qrels-v2.txt','r')

q_rel = {}

for line in rank_f:
    splits = line.split()
    key = splits[0]
    
    if key not in q_rel:
        q_rel[key] = 0
    else:
        if int(splits[3]) >0:
            
            q_rel[key] = q_rel[key]+1

print(q_rel)
recall = []
precision = []
for line in help_ff:
    non = 0
    jline = json.loads(line)
    tp = 0
    total = 0
    if jline['index'] in q_rel.keys():
        
        total = len(jline['candidates'].keys())
        for key in jline['candidates']:
            if jline['candidates'][key]['ans']:
                tp +=1
        if total>0:
            precision.append(tp/total)
        if q_rel[jline['index']]>0:
            recall.append(tp/q_rel[jline['index']])

print(sum(precision)/len(precision))
print(sum(recall)/len(recall))

           
