import json
import tagme
tagme.GCUBE_TOKEN = "e699faf6-94d2-4739-8d7c-097b7cd1be83-843339462"
import pandas as pd
import json
from elasticsearch import Elasticsearch
import itertools
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
current_directory = './'
import sys
import codecs
import unidecode
import random
import json
import re
import ast
import urllib.parse


def get_entities(query,conf=0.1):
  entities=set()
  dic = dict()
  annotations = tagme.annotate(query)
  try:
  # Print annotations with a score higher than 0.1
    for ann in annotations.get_annotations(conf):
        A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
        #dic[A] = [B,score]
        dic[B]=score
        entities.add(B)
  except:
      print("error")
  return dic,entities

def get_triples(query, qrel):
    triple_size = 5000
    triples = []

    _,query_entities = get_entities(query)
    for query_entity in query_entities:
        results = es.search(index="wiki-graph-index", body={
            "size": triple_size,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"title": query_entity}},
                        {"match": {"anchored_et": query_entity}}
                    ]
                }
            }
        })['hits']['hits']
        
        found = set()
        for entry in results:
            triple = {}
            subject = entry["_source"]["title"]
            relation = entry["_source"]["text"]
            anchored_ets = entry["_source"]["anchored_et"][0][0]
            score = entry["_score"]
            
            ans = False
            if subject.lower() in qrel:
                temp = subject 
                subject = anchored_ets
                anchored_ets = temp
            if anchored_ets.lower() in qrel:
                found.add(anchored_ets.lower())
                ans = True

        
        
        relation = re.sub(r'[^a-zA-Z0-9\s]', '', relation)
        triple = {"subject":subject,"relation":relation,"object":anchored_ets, "score":score,"ans":ans}
        
        
        triples.append(triple)
    
    return triples

def final_output(triples):
    results = {}
    
    for triple in triples:
        objectt =  triple["object"].lower() 
        if  objectt not in results.keys():
            results[objectt] =[]
        results[objectt].append(triple)

    return results


def retrive ():
    in_f = open("/data/dbpedia-v2/queries.txt","r")
    in_qrel = open("/data/dbpedia-v2/qrels.txt","r")
    out_f = open("/data/dbpedia-v2/test.json","w")
    qrel = {}
    for line in in_qrel:
        splits = line.split()
        query = urllib.parse.unquote(splits[0]).replace("enwiki:","").replace("tqa:","").lower()
        if query not in qrel.keys():
            qrel[query] = []
        entity = urllib.parse.unquote(splits[2]).replace("enwiki:","").replace("tqa:","").lower()
        score = splits[3]

        if int(score)>=1:
            qrel[query].append(entity)

    for line in in_f:
        triples = get_triples(line.strip(),qrel[line.strip().lower()])
        triples = final_output(triples)
        jobject = {"index":line.strip(),"question":line.strip(),"q_ets":[line.strip()],"qrel":qrel[line.strip().lower()],"candidates":triples}

        out_f.write(json.dumps(jobject)+"\n")
  
retrive()

        
        


    