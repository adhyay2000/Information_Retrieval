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
from spellchecker import SpellChecker

def main():
    while(1):
        print('----------------------------------------------------------------------------------------------------')
        print(" 1). Question 1 -> Normal search based on lnc.ltc scoring scheme.")
        print(" 2). Question 2.1 -> Improvement using spelling correction.")
        print(" 3). Question 2.2 -> Improvement using proximity search.")
        print(" 4). EXIT.")
        choice = (input("Enter your choice : "))
        if(choice == '1'):
            main_question_1()
        elif(choice == '2'):
            main_question_2_improvement_1()
        elif(choice == '3'):
            main_question_2_improvement_2()
        elif(choice == '4'):
            print('EXIT')
            break
        else:
            print("Please Enter valid choice.")


def remove_punctuation(txt):
    new_txt = "".join([c for c in txt  if c not in string.punctuation])
    return new_txt

## This function will give all the doc_ids which contains the query terms
def get_docs_id(query_string, answer_list): 
    query = query_string
    query_words = query.split()  
    try:
        len_of_posting_lists_corresponding_to_query_words = {}
        for word in query_words:
            len_of_posting_lists_corresponding_to_query_words[word] = len(final_d[word])
        len_of_posting_lists_corresponding_to_query_words = dict(sorted(len_of_posting_lists_corresponding_to_query_words.items(),key=lambda x:x[1]))
        print("Query_word : Num_of_docs_that_contains_query_term  =>" ,len_of_posting_lists_corresponding_to_query_words)

        query_words = [*len_of_posting_lists_corresponding_to_query_words.keys()]
        print("Query_words_list => ", query_words)
        
        temp_list = []
        for entry in final_d[query_words[0]]:
            temp_list.append(entry[0])

        for word in query_words:
            p1 = 0  ## Pointer to answer_list
            p2 = 0  ## Pointer to final_d[word]  list
            new_ans_list = []
            while(p1 < len(temp_list) and p2 < len(final_d[word])):
                if(temp_list[p1] == final_d[word][p2][0]):
                    new_ans_list.append(temp_list[p1])
                    p1 += 1
                    p2 += 1
                elif(temp_list[p1] < final_d[word][p2][0]):
                    p1 += 1
                else:
                    p2 += 1

            temp_list = new_ans_list
        answer_list.extend(temp_list)
    except Exception as e:
        print("Some word in the query is incorrect or it may not present in any of the document.")

## This function compute the score between the query and each docs and return the dictionary{doc_id : score, ...}
def compute_scores(query,doc_ids):
    punctuation_removed_file = open("IR_Assignment_2_punctuation_removed_file.txt","r", encoding="utf8")
    text = punctuation_removed_file.read()
    punctuation_removed_file.close()
    data = BeautifulSoup(text, 'html.parser')
    docs = data.select("doc")
    scores = {}
    for doc_id in doc_ids:
        doc_text = data.find(id=str(doc_id)).text
        terms_in_doc = doc_text.split()
        terms_in_query = query.split()
        query_terms_set = set(terms_in_query)
        doc_terms_set = set(terms_in_doc)

        ## Creating weights table for document
        query_terms_not_in_current_doc = query_terms_set.difference(doc_terms_set)
        doc_tf = {}
        for term in terms_in_doc:
            doc_tf[term] = doc_tf.get(term,0) + 1
        ## Calculating tf weights
        for term in doc_tf:
            doc_tf[term] = round(1 + np.log10(doc_tf[term]) , 3)
        for term in query_terms_not_in_current_doc:
            doc_tf[term] = 0
        ## Normalizing weights
        sqrs = np.sum(np.square(np.array([*doc_tf.values()])))
        normalizing_denominator = np.sqrt(sqrs)
        for term in doc_tf:
            doc_tf[term] = round(doc_tf[term]/normalizing_denominator , 3)



        ## Creating weights table for query
        current_doc_terms_not_in_query = doc_terms_set.difference(query_terms_set)
        query_tf = {}
        for term in terms_in_query:
            query_tf[term] = query_tf.get(term,0) + 1
        ## Calculating tf-idf weights
        for term in query_tf:
            query_tf[term] = round((1 + np.log10(query_tf[term]))*(np.log10(len(docs)/len(final_d[term]))) , 3)
        for term in current_doc_terms_not_in_query:
            query_tf[term] = 0
        ## Normalizing weights
        sqrs = np.sum(np.square(np.array([*query_tf.values()])))
        normalizing_denominator = np.sqrt(sqrs)
        for term in query_tf:
            query_tf[term] = round(query_tf[term]/normalizing_denominator , 3)

        ## Dot Product of normalized weights of corresponding terms in query and current document
        dot_prod = []
        for term in query_tf:
            dot_prod.append(doc_tf[term]*query_tf[term])

        scores[doc_id] = round(sum(dot_prod) , 5)
    scores = dict(sorted(scores.items(),key=lambda x:x[1],reverse=True))
    return scores

## Processing the query to extract the documents related to the query
def process_query(query):
    answer_list  = []
    ## Getting the list of doc_ids which contains the query terms
    get_docs_id(query, answer_list)
    ## Computing the score
    scores = compute_scores(query, answer_list)
    return scores

## This function prints the top k retrieved documents. By default it will print top 10 documents.
def print_retrieved_docs(scores,top_k_docs=10):   
    punctuation_removed_file = open("IR_Assignment_2_punctuation_removed_file.txt","r", encoding="utf8")
    text = punctuation_removed_file.read()
    punctuation_removed_file.close()
    data = BeautifulSoup(text, 'html.parser')
    try:
        for i in range(top_k_docs):
            print(i+1, "->   Doc_id------>", [*scores.keys()][i] ,"  |  ","Title---->",data.find(id=str([*scores.keys()][i]))["title"], "  |  ", "Score---->" ,[*scores.values()][i] )
    except Exception as e:
        print(f"Our search engine is able to retrieve only {i} documents.")

## This function is used to print the content of the retrieved top_k_docs
def print_docs_content(doc_ids,top_k_docs = 10):
    ## Extracting the original documents content
    file = open("IR_Assignment_2_file.txt", encoding="utf8")
    text = file.read()
    file.close()
    data = BeautifulSoup(text, 'html.parser')

    for index,doc_id in enumerate(doc_ids,start = 1): 
        if index > top_k_docs:
            break
        print(index,") ------------------------------------------------------------------------------------------------------")
        print()
        print("Doc_id    => ",data.find("doc",id=doc_id)["id"])
        print("Doc Title : ",data.find("doc",id=doc_id)["title"])
        print(data.find("doc",id=doc_id).text)
        print()

## Adding the spelling correction improvement
def spelling_correction(query):
    print("Actual(misspeled) Query words :",query.split())
    spell = SpellChecker()
    words = spell.split_words(query) 
    query = [spell.correction(word) for word in words]
    print("Modified Query words : ",query)
    query = " ".join(query)
    return query


def proximity_search(query,scores,top_k_docs):
    def relevance(query,document_txt):
        size=len(query)
        flag=1
        count=0
        for i in range(len(document_txt)-size+1):
            tmp=""
            flag=1
            for j in range(size):
                tmp += document_txt[j+i]
            for j in range(size):
                if(ord(tmp[j])^ord(query[j])!=0):
                    flag=0
                    break
            if(flag==1):
                count=count+1
        return count

    min_val = min(top_k_docs+10,len(scores))
    doc_id_relevance={}

    file = open("wiki_07.txt","r", encoding="utf8")
    text = file.read()
    file.close()
    data = BeautifulSoup(text, 'html.parser')
    docs = data.select("doc")

    d = {}
    for tag in docs:
        d[tag["id"]] = tag.text

    tmp_dict = {}
    for i in range(min_val):
        doc_id_relevance[[*scores.keys()][i]] = relevance(query,d[str([*scores.keys()][i])])
        if(relevance(query,d[str([*scores.keys()][i])]) == 0 ):
            tmp_dict[[*scores.keys()][i]] = [*scores.values()][i]
    doc_id_relevance = dict(sorted(doc_id_relevance.items(),key=lambda x:x[1],reverse=True))
    tmp_dict = dict(sorted(tmp_dict.items(),key = lambda x:x[1],reverse=True))

    count=0
    retrieved_docs = top_k_docs
    top_k_doc_ids = []
    for i in range(min(10,len(scores))):
        if(retrieved_docs ==0):
            break
        if([*doc_id_relevance.values()][i]!=0):
            top_k_doc_ids.append([*doc_id_relevance.keys()][i])
            count += 1
            print(i+1,"-> Doc_id---->",[*doc_id_relevance.keys()][i],"  |  Title---->",data.find(id=str([*doc_id_relevance.keys()][i]))["title"],"  |  ","Score---->",scores[[*doc_id_relevance.keys()][i]],"  |  Relevance---->",[*doc_id_relevance.values()][i])
            retrieved_docs -= 1
    count+=1
    for i in range(len(tmp_dict)):
        if(retrieved_docs ==0):
            break
        top_k_doc_ids.append([*tmp_dict.keys()][i])
        print(i+count,"-> Doc_id---->",[*tmp_dict.keys()][i],"  |  Title---->",data.find(id=str([*tmp_dict.keys()][i]))["title"]," | Score---->",[*tmp_dict.values()][i],"  |  Relevance----> 0")
        retrieved_docs -= 1
    return top_k_doc_ids

def main_question_1():
    query = input("Write your query : ")
    query = remove_punctuation(query.lower())
    top_k_docs = int(input("No. of documents you want to retrieve : "))
    print('processing query...')
    scores = process_query(query)
    print()
    print("Top retrieved docs are : ")
    print_retrieved_docs(scores,top_k_docs) ## Printing the doc_id, title, score
    ## Uncomment the below line  of code if you want to print the content of the code also.
    #doc_ids = [*scores.keys()]
    #print_docs_content(doc_ids,top_k_docs)


def main_question_2_improvement_1():
    query = input("Write your query : ")
    query = remove_punctuation(query.lower())
    top_k_docs = int(input("No. of documents you want to retrieve : "))
    query = spelling_correction(query)
    scores = process_query(query)
    print()
    print("Top retrieved docs are : ")
    print_retrieved_docs(scores,top_k_docs) ## Printing the doc_id, title, score
    ## Uncomment the below line  of code if you want to print the content of the code also.
    #doc_ids = [*scores.keys()]
    #print_docs_content(doc_ids,top_k_docs)


def main_question_2_improvement_2():
    query = input("Write your query : ")
    query = remove_punctuation(query.lower())
    top_k_docs = int(input("No. of documents you want to retrieve : "))
    scores = process_query(query)
    print("Top retrieved docs are : ")
    top_K_doc_ids = proximity_search(query,scores,top_k_docs)
    ## Uncomment the below line  of code if you want to print the content of the code also.
    #print_docs_content(top_K_doc_ids,top_k_docs)

#############################################################################################################

# Loading the saved dictionary {"word" : [[doc_id,frequency] , .......]} from the saved text file
file = open("IR_Assignment_2_sorted_inverted_index.txt","r", encoding="utf8")
final_d = eval(file.read())
file.close()
main()
