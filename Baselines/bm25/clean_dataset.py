import json

in_f = open('/home/jparastoo/downloads/dbpedia-v2/bm25/fsdm-made_data_all.json','r')
out_f = open('/home/jparastoo/downloads/dbpedia-v2/bm25/fsdm-final.json','w')

for line in in_f:
    jline =json.loads(line)

    new_candidates = {}
    for candidate in jline['candidates']:
        if 'subject'  in jline['candidates'][candidate]:
            new_candidates[candidate] = jline['candidates'][candidate]
    
    jline['candidates'] = new_candidates
    out_f.write(json.dumps(jline)+"\n")
out_f.close()