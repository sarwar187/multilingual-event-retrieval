import numpy as np 
import pandas as pd
import json 
import re
import os
from code.qproc.indri_query_preparation import sample_queries_for_types
from code.qproc.indri_query_preparation import sample_queries_for_combined_types
import time
import logging

logger = logging.getLogger('better')
hdlr = logging.FileHandler('better.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

def main():
    config = json.load(open("code/config/unsupervised_lm_config.json"))
    data_directory = config["data"]
    src_lang = config["src_lang"]
    trg_lang = config["trg_lang"]
    approach = config["approach"]
    query2id_file = config["query_to_id_file"]
    translation = config["translation"]
    doc_dir = config["index_dir"] #we are naming it as document directory because we do not build an index for unsupervised lm 
    #query is event type
    query2id = json.load(open(os.path.join(data_directory, src_lang, "queries", query2id_file)))
    df = pd.read_csv(os.path.join(data_directory, src_lang, "queries", translation, src_lang + "_translations.csv"), sep="\t")
    df['sentence_translation'].fillna("dummy", inplace=True)
    df['trigger_translation'].fillna("dummy", inplace=True)

    approaches = ["unsupervised_lm"]
    representations = ["bert"]
    query_types = ["combined"]

    run_id = 1

    for approach in approaches:
        for representation in representations:
            for query_type in query_types:
                for num_examples in range(1,11):
                    if query_type == "combined":
                        query2text, query2triggers = sample_queries_for_combined_types(df, num_sentences = num_examples)
                    else:    
                        query2text = sample_queries_for_types(df, query_type, num_sentences = num_examples)                    
                    query_dict = {}
                    id2text = {}
                    id2triggers = {} 

                    for key in query2text.keys():
                        text = query2text[key] #query2text comes up with dictionary entries like
                        id = query2id[key]
                        id2text[id] = text.strip()
                        if query_type == "combined":
                            trigger = query2triggers[key]
                            id2triggers[id] = trigger.strip()
                        print(text)
                        print(trigger)
                        assert(len(text.strip().split("\t")) == len(trigger.strip().split("\t")))
                        print("{} \t {}".format(key, text))

                    query_dict["run_id"] = run_id
                    query_dict["trg_lang"] = trg_lang
                    query_dict["doc_dir"] = doc_dir #here we have a document directory instead of index directory because we do not need indexing for 16000 documents.
                    query_dict["queries"] = id2text
                    query_dict["triggers"] = id2triggers
                    timestr = time.strftime("%Y%m%d-%H%M%S")
                    os.makedirs(os.path.join(data_directory, src_lang, approach + "_queries", representation, query_type), exist_ok=True)
                    query_file = open(os.path.join(data_directory, src_lang, approach + "_queries", representation, query_type , timestr + "_" + str(num_examples) + "_query.json"), "w")
                    json.dump(query_dict, query_file)
                    run_id+=1

if __name__ == "__main__":
    main()
