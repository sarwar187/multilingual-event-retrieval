import numpy as np 
import pandas as pd
import json 
import re
import os
import sys 
from code.search.unsupervised_lm import UnsupervisedLM
from code.eval.search_and_eval import eval
from code.util.doc_util import get_event_windows_simple

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



def main():
    config = json.load(open("code/config/unsupervised_lm_config.json"))
    data_directory = config["data"]
    src_lang = config["src_lang"]
    trg_lang = config["trg_lang"]
    approach = config["approach"]
    doc_dir = config["index_dir"] #we are naming it as document directory because we do not build an index for unsupervised lm 
    representation = config["representation"]
    query_type = "combined"
    
    mode = "search"
    
    unsupervised = UnsupervisedLM(config)
    documents = parse_indri_documents(os.path.join(data_directory, trg_lang, doc_dir, "docs.xml"))     
    document_contents, document_windows = chunck_documents(documents)   

    documents_embeddings = unsupervised.bert_representation(document_contents)
    logger.info("Documents embeddings have been loaded")        
    os.makedirs(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach, representation, query_type), exist_ok=True)
    result_file = open(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach,representation, query_type, "output.res"), "w")
    for num_examples in range(1,31):                
        if mode=="search":
            logger.info("Searching the {} number of examples".format(num_examples))
            query_dict = json.load(open(os.path.join(data_directory, src_lang, approach + "_queries", representation, query_type , str(num_examples) + "_query.json")))                  
            result_string = unsupervised.search_combined(query_dict, document_contents, documents_embeddings, document_windows, documents)
            os.makedirs(os.path.join(data_directory, trg_lang, "runs", approach, representation, query_type), exist_ok=True)
            run_file = open(os.path.join(data_directory, trg_lang, "runs", approach, representation, query_type, str(num_examples) + ".run"), "w")
            run_file.write(result_string)
            run_file.close()
            run_file_path = os.path.join(data_directory, trg_lang, "runs", approach, representation, query_type, str(num_examples) + ".run")
            qrel_file_path = os.path.join(data_directory, trg_lang, "queries", "qrels." + trg_lang + "_events.txt") 
            p5, p10, p20, mAP, rprec = eval(qrel_file_path, run_file_path)
            logger.info("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))            
            result_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))            
        else:            
            run_file_path = os.path.join(data_directory, trg_lang, "runs", approach, representation, query_type, str(num_examples) + ".run")
            qrel_file_path = os.path.join(data_directory, trg_lang, "queries", "qrels." + trg_lang + "_events.txt") 
            p5, p10, p20, mAP, rprec = eval(qrel_file_path, run_file_path)
            print("{}\t{}\t{}\t{}\t{}\t{}".format(num_examples, p5, p10, p20, mAP, rprec))
            
    result_file.close()

if __name__ == "__main__":
    main()
