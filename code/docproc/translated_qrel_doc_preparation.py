import json
import os
import pandas as pd
import os.path


def create_query_to_id_dictionary(df_raw, config):
    df_raw['sentence_translation'].fillna("dummy", inplace=True)
    df_raw['trigger_translation'].fillna("dummy", inplace=True)

    query_id = 1
    query2id = {}

    for index, row in df_raw.iterrows():
        event_type = row['Event_Type'].strip()
        if event_type not in query2id:
            query2id[event_type] = query_id
            query_id+=1

    return query2id


def create_trec_data(query2id, df_raw, config):
    df_raw['sentence_translation'].fillna("dummy", inplace=True)
    df_raw['trigger_translation'].fillna("dummy", inplace=True)

    docno = 1
    qrel_file = open(os.path.join(config["data"], config["trg_lang"], config["query_dir"],
                                  "qrels." + config["trg_lang"] + "_events.txt"), "w")
    indri_doc_file = open(os.path.join(config["data"], config["trg_lang"], config["raw_index_dir"], "docs.xml"), "w")
    
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
df_raw_dir = os.path.join(config["data"], config["trg_lang"], config["query_dir"], config["translation"], config["trg_lang"] + "_translations.csv")
df_raw = pd.read_csv(open(df_raw_dir), sep="\t")
query2id_filepath = os.path.join(config["data"], config["query_to_id_file"])
if os.path.exists(query2id_filepath):
    query2id_file = open(query2id_filepath)
    query2id = json.load(query2id_file)
else:
    query2id = create_query_to_id_dictionary(df_raw, config)
    json.dump(query2id, open(query2id_filepath, "w"))

create_trec_data(query2id, df_raw, config)
