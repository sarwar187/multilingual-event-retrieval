import json
import os
import pandas as pd


def create_trec_data(query2id, df_raw, config):
    df_raw['sentence_translation'].fillna("dummy", inplace=True)
    df_raw['trigger_translation'].fillna("dummy", inplace=True)

    docno = 1
    qrel_file = open(os.path.join(config["data"], config["src_lang"], config["query_dir"],
                                  "qrels." + config["src_lang"] + "_events.txt"), "w")
    indri_doc_file = open(os.path.join(config["data"], config["src_lang"], config["raw_index_dir"], "docs.xml"), "w")
    
    for index, row in df_raw.iterrows():
        text = row['sentence_translation']
        trec_doc = "<DOC>\n"
        trec_doc += "<DOCNO>" + str(docno) + "</DOCNO>\n"
        trec_doc += "<TEXT>" + text + "</TEXT>\n"
        trec_doc += "</DOC>\n"

        if len(row['Event_Type']) > 0:
            qid = query2id[row['Event_Type']]
            qrel_file.write(str(qid) + " 0 " + str(docno) + " " + str(1) + "\n")

        docno += 1
        indri_doc_file.write(trec_doc)
    qrel_file.close()
    indri_doc_file.close()


config = json.load(open("code/config/basic_config_ace.json"))
df_raw_dir = os.path.join(config["data"], config["src_lang"], config["query_dir"], config["translation"], config["src_lang"] + "_translations.csv")
df_raw = pd.read_csv(open(df_raw_dir), sep="\t")
query2id_file = open(os.path.join(config["data"], config["query_to_id_file"]))
query2id = json.load(query2id_file)
create_trec_data(query2id, df_raw, config)
