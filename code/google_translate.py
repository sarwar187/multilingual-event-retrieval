import pandas as pd 
import os 
from googletrans import Translator
from json.decoder import JSONDecodeError
def google_translate(text):
    print(text)
    translator = Translator()
    translation = translator.translate(text, src='ar', dest='en')
    print(translation)
    print(type(translation))

    return translation.text
    

base = os.getcwd()
print(base)
arabic_queries = os.path.join(base, "small_data/ace/arabic/arabic_query.csv")
sentence_translations_dir = "small_data/ace/arabic/translations/sentences/"
trigger_translations_dir = "small_data/ace/arabic/triggers/"

#getting sentence translations done
sentence_translated = set()
files = os.listdir(os.path.join(base, sentence_translations_dir))

for f in files: 
    sentence_translated.add(int(f.split(".")[0]))

print(sentence_translated)

df = pd.read_csv(arabic_queries, sep="\t", header=None)
print(df.head())
print(len(df[3]))

translation_list = []

count = 0
        

# for arabic_sentence in df[3][1:]:
#     if count in sentence_translated:
#         count+=1
#         continue
#     else:
#         print(arabic_sentence)
#         try:
#             translation = google_translate(arabic_sentence)
#         except JSONDecodeError:
#             print("error happened for count {}".format(count))
#             count+=1
#             continue  
#         translation = " ".join(translation.split()[1:])
#         #print(df[3][1])
#         f = open(os.path.join(sentence_translations_dir, str(count)  + ".txt"), "w")
#         f.write(translation + "\n")
#     #print(translation)
    
count = 1
storage = []
for arabic_sentence in df[3][1:]:
    storage.append(arabic_sentence)
    translation = google_translate(arabic_sentence)
    print(type(translation))
    break
    if count%25 == 0:
        f = open(os.path.join(sentence_translations_dir, str(count - 25) + "-" + str(count)  + ".txt"), "w")
        for item in storage: 
            f.write(item + "\n")
        f.close()
        storage = []
        count+=1
    count+=1
    #print(translation)
    
   
