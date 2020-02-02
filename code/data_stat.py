# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import os 
import pandas as pd 
import matplotlib.pyplot as plt
        
def json_to_event_type_distribution(dict_raw):
    """[summary]
    
    Arguments:
        dict_raw {[dictionary]} -- [a dictionary containing event types and sentences describing them with other information]
    
    Returns:
        [type] -- [a dictionary where event type is a key and sentences are values]
    """
    event_types = {}
    for sentence in dict_raw:
        if len(sentence["golden-event-mentions"]) > 0: 
            for i, mention in enumerate(sentence["golden-event-mentions"]):
                event_types.setdefault(sentence["golden-event-mentions"][i]["event_type"], [])
                l = [sentence["sentence"], sentence["golden-event-mentions"][i]["trigger"]["text"], sentence["golden-event-mentions"][i]["trigger"]["start"], sentence["golden-event-mentions"][i]["trigger"]["end"]]
                event_types[sentence["golden-event-mentions"][i]["event_type"]].append(l)

    return event_types

def event_type_to_distribution(event_types, result_directory, language):
    """[This function takes event distritbution and write stats in the result directory]
    
    Arguments:
        event_types {[dictionary]} -- [a dictionary where event types are keys and sentences are values]
        result_directory {[type]} -- [the directory where dataset statistics are located, its language specific]
                                  -- the result directory has data and figures subdirectory 
    """
    data = []
    count = 0
    for key in event_types: 
        for sentence in event_types[key]:
            data.append([key, len(event_types[key]), sentence[0], sentence[1], sentence[2], sentence[3]])

    df = pd.DataFrame(data, columns=["Event_Type", "Count", "Sample_sentence", "Trigger", "Start", "End"])
    
    df.to_csv(os.path.join(result_directory, "data", language + "_query.csv"), sep="\t", encoding="utf-8", index=True)
    
    values = [] #in same order as traversing keys
    keys = [] #also needed to preserve order
    count = 0
    for key in event_types.keys():
        keys.append(count)
        count+=1
        values.append(int(len(event_types[key])))
    plt.bar(keys, values, color='g')
    plt.savefig(os.path.join(result_directory, "figures", language + "_event_type_distribution.png"))
    

def main():
    """
    [This function takes a config file and prints statistics for different languages in ACE 2005 dataset]
    """
    config = json.load(open("code/config/basic_config_ace.json"))
    print(os.getcwd())
    data_directory = config["data"]

    etype2count = {}
    
    languages = os.listdir(data_directory)
    language2event2type = {}
    
    #opening the raw files for each language to pull of some dataset statistics 
    for language in languages:
        result_directory = os.path.join(config["data"], language, "results")
        raw_path = open(os.path.join(config["data"], language, "raw", "raw.json"))
        dict_raw = json.load(raw_path)
        print("number of sentence in {} is {}".format(language, len(dict_raw)))
        #print(dict_raw[0])
        etype2sentences = json_to_event_type_distribution(dict_raw)
        language2event2type[language] = etype2sentences
        
        for etype in etype2sentences:
            etype2count.setdefault(etype, [])
    
        #this function will write dataset stats into the result directory
        event_type_to_distribution(etype2sentences, result_directory, language)

    for language in languages:
        etype2sentences = language2event2type[language]
        
        for etype in etype2count.keys():
            if etype in etype2sentences:
                etype2count[etype].append(len(etype2sentences[etype]))
            else:
                etype2count[etype].append(0)
    
    df = pd.DataFrame.from_dict(etype2count, orient='index', columns= languages)
    df.to_csv(os.path.join(config["data"], "..", "event_type_across_languages.csv"), sep='\t', encoding='utf-8')
    
if __name__ == "__main__":
    main()
    



# %%
