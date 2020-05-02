import numpy as np 
import pandas as pd
import json 
import re
import os
import sys 
from code.search.unsupervised_lm import UnsupervisedLM
from code.eval.search_and_eval import eval
from code.util.doc_util import get_event_windows_simple
import time 


import logging
logger = logging.getLogger('better')
hdlr = logging.FileHandler('better.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)



def parse_indri_documents(indri_file):    
    documents = []
    for line in open(indri_file):
        if line.startswith("<TEXT>"):
            line = line.strip()
            line = line.replace("<TEXT>", "")
            line = line.replace("</TEXT>", "")
            documents.append(line)
    return documents 

def chunck_documents(documents):
    document_windows = []
    document_contents = []
    begin = 0
    docno = 1
    for document in documents:         
        windows, _ = get_event_windows_simple(document)
        windows.append(document)
        document_contents.extend(windows)
        end = begin + len(windows)
        document_windows.append((begin, end))
        begin = end
        print("done with document number {}".format(docno))
        docno+=1
    return document_contents, document_windows

def read_run_file(ql_run_file):
    run_dict = {}
    for line in open(ql_run_file):
        line_splitted = line.strip().split()
        query_id = line_splitted[0].strip()
        document_id = line_splitted[2].strip()
        rank = line_splitted[3].strip()
        score = line_splitted[4].strip()
        run_dict.setdefault(query_id, [])
        run_dict[query_id].append((document_id, rank, score))
    return run_dict


def load_queries(query_directory):
    files = os.listdir(query_directory)
    query_files = []
    for f in files:
        print(f)
        name = f.split(".")[0]
        print(name)
        name_splitted = name.split("_")
        timestr = name_splitted[0]
        num_examples = name_splitted[1]
        file_name = f
        query_files.append((timestr, num_examples, file_name))
    return query_files

def main():
    config = json.load(open("code/config/unsupervised_lm_config.json"))
    data_directory = config["data"]
    src_lang = config["src_lang"]
    trg_lang = config["trg_lang"]
    approach = config["approach"]
    doc_dir = config["index_dir"] #we are naming it as document directory because we do not build an index for unsupervised lm 
    representation = config["representation"]
    query_type = "sentences"
    mode = "search"        
    
    unsupervised = UnsupervisedLM(config)
    documents = parse_indri_documents(os.path.join(data_directory, trg_lang, doc_dir, "docs.xml"))     
    documents_embeddings = unsupervised.bert_representation(documents)
    #document_contents, document_windows = chunck_documents(documents)   
    #documents_embeddings = unsupervised.bert_representation(document_contents)
    logger.info("Documents embeddings have been loaded")        
    os.makedirs(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach, representation, query_type), exist_ok=True)
            
    #for num_examples in range(1,11):         
    query_dir = os.path.join(data_directory, src_lang, "universal_queries") 
    query_files = load_queries(query_dir)
    result_file = open(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach,representation, query_type, "output.res"), "w")
            
    for query_file in query_files:
        if mode=="search":
            timestr = query_file[0]
            num_examples = int(query_file[1])
            file_name = query_file[2]
            logger.info("Searching the {} number of examples".format(num_examples))
            #TODO: CHANGE THE FOLLOWING LINE TO ADOPT TO DIFFERENT PARAMETERS
            query_dict = json.load(open(os.path.join(data_directory, src_lang, "universal_queries", file_name)))                  
            result_string = unsupervised.search(query_dict, documents, documents_embeddings)
            os.makedirs(os.path.join(data_directory, trg_lang, "runs", src_lang, approach, representation, query_type), exist_ok=True)
            run_file = open(os.path.join(data_directory, trg_lang, "runs", src_lang, approach, representation, query_type, timestr + "_" + str(num_examples)  + ".run"), "w")
            run_file.write(result_string)
            run_file.close()
            run_file_path = os.path.join(data_directory, trg_lang, "runs", src_lang, approach, representation, query_type, timestr + "_" + str(num_examples) + ".run")
            qrel_file_path = os.path.join(data_directory, trg_lang, "queries", "qrels." + trg_lang + "_events.txt") 
            p5, p10, p20, mAP, rprec = eval(qrel_file_path, run_file_path)
            logger.info("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))            
            result_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))            
        else:            
            run_file_path = os.path.join(data_directory, trg_lang, "runs", src_lang, approach, representation, query_type, timestr + "_" + str(num_examples) + ".run")
            qrel_file_path = os.path.join(data_directory, trg_lang, "queries", "qrels." + trg_lang + "_events.txt") 
            p5, p10, p20, mAP, rprec = eval(qrel_file_path, run_file_path)
            print("{}\t{}\t{}\t{}\t{}\t{}".format(num_examples, p5, p10, p20, mAP, rprec))
            
    result_file.close()

if __name__ == "__main__":
    main()
