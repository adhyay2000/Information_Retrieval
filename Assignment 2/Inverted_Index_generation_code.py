# Name:  Darshan Agrawal, Aditya Upadhyay
# Student ID: 2017A7PS0233P , 2017A7PS0083P 
# BITS Email: f20170233@pilani.bits-pilani.ac.in , f20170083@pilani.bits-pilani.ac.in
# Wikipedia file used: AN/wiki_07
from bs4 import BeautifulSoup
import requests
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
import string
#############################################################################################################
## This block of code is extracting the the text data from the corpus by removing the tags.
file = open("wiki_07.txt","r", encoding="utf8")
text = file.read()
file.close()
data = BeautifulSoup(text, 'html.parser')
docs = data.select("doc")

d = {}
for tag in docs:
    d[tag["id"]] = [tag["title"],tag.text]

print('Done preprocessing on corpus....')
file = open("IR_Assignment_2_file.txt","w", encoding="utf8")
for doc_id,ls in d.items():
    file.write('<doc id="')
    file.write(doc_id)
    file.write('" title="')
    file.write(ls[0])
    file.write('" >\n')
    file.write(ls[1])
    file.write("</doc>\n")
file.close()

file = open("IR_Assignment_2_file.txt", encoding="utf8")
text = file.read()
file.close()

data = BeautifulSoup(text, 'html.parser')
docs = data.select("doc")

#############################################################################################################
## This block of the code is used for tokenizing and removing punctuations from the file "IR_Assignment_2_file.txt" .
# file = open("IR_Assignment_2_file.txt", encoding="utf8").read()
def remove_punctuation(txt):
    new_txt = "".join([c for c in txt  if c not in string.punctuation])
    return new_txt

for i in range(len(d)):
    punctuation_removed_doc = remove_punctuation([*d.values()][i][1])
    d[[*d.keys()][i]] = [[*d.values()][i][0],punctuation_removed_doc]

file = open("IR_Assignment_2_punctuation_removed_file.txt","w", encoding="utf8")
for doc_id,ls in d.items():
    file.write('<doc id="')
    file.write(doc_id)
    file.write('" title="')
    file.write(ls[0])
    file.write('" >\n')
    file.write(ls[1])
    file.write("</doc>\n")
file.close()
print('Removed punctuations from the corpus....')
#############################################################################################################
## This block of code makes the list of words with their corresponding doc_ids & frequencies in the docs
punctuation_removed_file = open("IR_Assignment_2_punctuation_removed_file.txt","r", encoding="utf8")
text = punctuation_removed_file.read()
punctuation_removed_file.close()

data = BeautifulSoup(text, 'html.parser')
docs = data.select("doc")
ls = []
for doc in docs:
    tokenized_words = word_tokenize(doc.text)
    tokenized_words = [word.lower() for word in tokenized_words]
    dct = dict()
    for word in tokenized_words:
        if dct.get(word,0) == 0:
            dct[word] = [doc['id'] , 1]
        else:
            dct[word][1] += 1            
    ls.append(dct)
#print(ls)

#############################################################################################################
## This block of code was creating the inverted index for the documents. So, it will take some time for its execution.
print('Creating inverted index...')
set1 = set(ls[0])
final_set = set()
final_d = {}
for word in set1:
    final_d[word] = [[int(docs[0]['id']), ls[0][word][1]]]

for i in range(1,len(docs)):
    set2 = set(ls[i])
    intersection_set = set1.intersection(set2)
    left_part = set1.difference(set2)
    right_part = set2.difference(set1) 
    final_set = intersection_set.union(left_part.union(right_part))
    set1 = final_set    
    for word in right_part:
        final_d[word] = [[int(docs[i]['id']) , ls[i][word][1]]]
    for word in intersection_set:
        final_d[word].append([int(docs[i]['id']) , ls[i][word][1]])
## Sorting the documents on the basis of doc_id
for word in final_d:
    final_d[word] = sorted(final_d[word])

## Saving the {"word" : [[doc_id,frequency] , .......]}  dictionary into the text file for later usage
print('Saving the inverted index for future reference...')
file = open("IR_Assignment_2_sorted_inverted_index.txt","w", encoding="utf8")
file.write(str(final_d))
file.close()
