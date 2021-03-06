import json 
import os 

def create_query_to_id_dictionary():
    dict_raw = json.load(open("small_data/ace/english/raw/raw.json"))
    print(len(dict_raw))

    query_id = 1 
    query2id = {} 
    
    for item in dict_raw:
        if len(item["golden-event-mentions"]) > 0: 
            for i, mention in enumerate(item["golden-event-mentions"]):
                if item["golden-event-mentions"][i]["event_type"] not in query2id:
                    query2id[item["golden-event-mentions"][i]["event_type"]] = query_id
                    query_id+=1
    
    return query2id, dict_raw

def create_trec_data(query2id, dict_raw):
    docno = 1
    qrel_file = open("small_data/ace/english/queries/qrels.english_events.txt", "w")
    indri_doc_file = open("small_data/ace/english/indri_raw/docs.xml", "w")
    query2id_file = open("small_data/ace/query2id.json", "w")
    
    for item in dict_raw:
        text = item['sentence']
        trec_doc = "<DOC>\n"
        trec_doc += "<DOCNO>" + str(docno) + "</DOCNO>\n"
        trec_doc += "<TEXT>" + text + "</TEXT>\n"
        trec_doc += "</DOC>\n"
        
        if len(item["golden-event-mentions"]) > 0: 
            for i, mention in enumerate(item["golden-event-mentions"]):
                qid = query2id[item["golden-event-mentions"][i]["event_type"]]
                qrel_file.write(str(qid) + " 0 " + str(docno) + " " + str(1) + "\n")        
        
        docno+=1        
        indri_doc_file.write(trec_doc)    
    qrel_file.close()
    json.dump(query2id, query2id_file)
    indri_doc_file.close()
    

query2id, dict_raw  = create_query_to_id_dictionary()
create_trec_data(query2id, dict_raw)
