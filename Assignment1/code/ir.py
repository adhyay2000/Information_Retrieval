#!/usr/bin/env python
# coding: utf-8

# ## Text Preprocessing IR Assignment

# In[1]:

# THIS function checks for if the given string is in english or not
def isEnglish(str1):
    for i in range(len(str1)):
        if not((str1[i]>='A' and str1[i]<='Z') or (str1[i]>='a' and str1[i]<='z')):
            return False
    return True


# In[2]:

# all the import statement are written here for reference
import nltk
import re
import matplotlib.pyplot as plt
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.util import trigrams
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
nltk.download('wordnet')
nltk.download('punkt')


# In[3]:

# function to plot unigrams from a given list l
def unigram_plot(l):
    dicitonary_unigram={}
    dicitonary_unigram=dict.fromkeys(l,0)
    for i in range(0,len(l)):
        dicitonary_unigram[l[i]]+=1;
    plus=0
    for i in dicitonary_unigram.values():
        plus+=i
    print("total unique unigram are:",end="")
    print(len(dicitonary_unigram.keys()))
    
    #the sorted function takes argument the dict_items, key and reverse flag
    # the lambda x:x[1] means sort by taking the frequency as the key
    dicitonary_unigram = dict(sorted(dicitonary_unigram.items(),key=lambda x:x[1],reverse=True))    
    count=0
    pdf = 0.0
    for key,i in dicitonary_unigram.items():
        pdf += i/plus
        count+=1
        if(pdf > 0.9):
            break
    print("total unigrams are required to cover the 90% of the complete corpus:",end="")
    print(count)
    print(pdf)
    # Due to H/W constraint on the system, a certain representative threshold values are taken and plotted 
    threshold = 200
    d_dictionary=dicitonary_unigram.items()
    for key in dicitonary_unigram.copy():
        if(dicitonary_unigram[key]<threshold):
            dicitonary_unigram.pop(key)
    lists = dicitonary_unigram
    keys_unigram,values_unigram=lists.keys(),lists.values()
    print("total unigrams taken for plotting purpose:",end="")
    print(len(keys_unigram))
    fig = plt.figure()
    plt.loglog(tuple(keys_unigram),tuple(values_unigram),color='g')
    plt.xticks(range(len(keys_unigram)),keys_unigram,rotation=90)
    plt.xlabel('unigram')
    plt.ylabel('unigram count')
    plt.xscale('log')
    plt.show()
    fig.savefig('unigram')

# In[4]:

#function to plot bigrams distribution
def bigram_plot(l):
    list_bigrams=list(bigrams(l))
    dictionary_bigram={}
    dicitonary_bigram=dict.fromkeys(list_bigrams,0)

    for i in range(len(list_bigrams)):
        dictionary_bigram[list_bigrams[i]]=0
    for i in range(len(list_bigrams)):
        dictionary_bigram[list_bigrams[i]]+=1
    plus=0
    for i in dictionary_bigram.values():
        plus+=i
    print("total unique bigram present:",end="")
    print(len(dictionary_bigram.keys()))
    
    dictionary_bigram = dict(sorted(dictionary_bigram.items(),key=lambda x:x[1],reverse=True))
    count=0
    pdf = 0.0
    for key,i in dictionary_bigram.items():
        pdf += i/plus
        count+=1
        if(pdf > 0.8):
            break
    print("total bigrams are required to cover the 80% of the complete corpus:",end="")
    print(count)
    print(pdf)
    
    threshold = 100
    for key in dictionary_bigram.copy():
        if(dictionary_bigram[key]<threshold):
            dictionary_bigram.pop(key)
    print(len(dictionary_bigram))
    keys_bigram,values_bigram=dictionary_bigram.keys(),dictionary_bigram.values()
    keys_bigram = list(keys_bigram)
    ls = []
    #bigram converted to space seprated string
    for i in keys_bigram:
        t = ' '.join(i)
        ls.append(t)
    print("total bigrams taken for plotting purpose:",end="")
    print(len(ls))
    plt.loglog(tuple(ls),tuple(values_bigram),color='g')
    plt.xticks(range(len(ls)),ls,rotation=90)
    plt.xlabel('bigram')
    plt.ylabel('bigram count')
    plt.xscale('log')
    plt.show()
    plt.savefig('bigram')

# In[5]:


def trigram_plot(l):
    list_trigrams=list(trigrams(l))
    dictionary_trigram={}
    for i in range(len(list_trigrams)):
        dictionary_trigram[list_trigrams[i]]=0
    for i in range(len(list_trigrams)):
        dictionary_trigram[list_trigrams[i]]+=1
    plus=0
    for i in dictionary_trigram.values():
        plus+=i    
    print("total unique trigram are:",end="")
    print(plus)
    
    dictionary_trigram = dict(sorted(dictionary_trigram.items(),key=lambda x:x[1],reverse=True))
    count=0
    pdf = 0.0
    for key,i in dictionary_trigram.items():
        pdf += i/plus
        count+=1
        if(pdf > 0.7):
            break
    print("total trigrams are required to cover the 70% of the complete corpus:",end="")
    print(count)
    print(pdf)

    
    threshold =45
    for key in dictionary_trigram.copy():
        if(dictionary_trigram[key]<threshold):
            dictionary_trigram.pop(key)
    keys_trigram,values_trigram=dictionary_trigram.keys(),dictionary_trigram.values()
    keys_trigram = list(keys_trigram)
    ls = []
    for i in keys_trigram:
        t = ' '.join(i)
        ls.append(t)
    print("total trigrams taken for plotting purpose:",end="")
    print(len(ls))
    plt.loglog(tuple(ls),tuple(values_trigram),color='g')
    plt.xticks(range(len(ls)),ls,rotation=90)
    plt.xlabel('trigram')
    plt.ylabel('trigram count')
    plt.xscale('log')
    plt.show()
    plt.savefig('trigram')
# In[6]:


with open('output.txt') as f:
    s=f.read()
translate_table = dict((ord(char), None) for char in string.punctuation)   
s=s.translate(translate_table)
lst=word_tokenize(s)
for i in lst:
  i=re.sub("[^A-Za-z]","",i.strip())
l = [i.lower() for i in lst if isEnglish(i)]
unigram_plot(l)
bigram_plot(l)
trigram_plot(l)


# In[7]:


stemmer = SnowballStemmer("english",ignore_stopwords=True)
stemmed_list=[stemmer.stem(words) for words in l]


# In[8]:


unigram_plot(stemmed_list)
bigram_plot(stemmed_list)
trigram_plot(stemmed_list)


# In[9]:


wnl = WordNetLemmatizer()
lemmatized_words=[wnl.lemmatize(words) for words in l]


# In[10]:


unigram_plot(lemmatized_words)
bigram_plot(lemmatized_words)
trigram_plot(lemmatized_words)


# In[28]:

import decimal 
from decimal import Decimal
decimal.getcontext().prec = 6
list_bigrams=list(bigrams(l))
dictionary_bigram={}
dictionary_bigram=dict.fromkeys(list_bigrams,0)
for i in range(len(list_bigrams)):
    dictionary_bigram[list_bigrams[i]]=0
for i in range(len(list_bigrams)):
    dictionary_bigram[list_bigrams[i]]+=1
dictionary_unigram={}
dictionary_unigram=dict.fromkeys(l,0)
for i in range(0,len(l)):
    dictionary_unigram[l[i]]+=1;
count=0
chisq={}

#way 1:
for key,frequency in dictionary_bigram.copy().items():
    words = list(key)
    both_present = frequency
    only_first = dictionary_unigram[words[0]]
    only_second = dictionary_unigram[words[1]]
    first_not_second = only_first - both_present
    second_not_first = only_second - both_present
    both_absent = len(dictionary_unigram)-(second_not_first+first_not_second)
    chisquare_num = len(dictionary_bigram)*(both_present*both_absent - first_not_second*second_not_first)**2
    chisquare_den = (both_present+first_not_second)*(both_present+second_not_first)*(both_absent+first_not_second)*(both_absent+second_not_first)
    value = decimal.Decimal(chisquare_num)/decimal.Decimal(chisquare_den)    
    if(value>3.84):
        chisq[key]=value

#way 2:
# for key,frequency in dictionary_bigram.copy().items():
#     words = list(key)
#     both_present = frequency
#     only_first = dictionary_unigram[words[0]]
#     only_second = dictionary_unigram[words[1]]
#     first_not_second = only_first - both_present
#     second_not_first = only_second - both_present
#     both_absent = len(dictionary_unigram)-(second_not_first+first_not_second)
#     mat1 = np.array([[both_present,second_not_first],[first_not_second,both_absent]])
#     row1_sum = np.sum(mat1[0,:])
#     row2_sum = np.sum(mat1[1,:])
#     col1_sum = np.sum(mat1[:,0])
#     col2_sum = np.sum(mat1[:,1])
#     total_sum = np.sum(mat1)    
#     mat2 = np.array([[row1_sum*col1_sum,row1_sum*col2_sum],[row2_sum*col1_sum,row2_sum*col2_sum]])/total_sum
#     if(np.sum(mat2)!=0):
#         chi_square = np.round(np.sum((((mat1-mat2)**2)/mat2)),3)
#         if chi_square>3.84:
#             chisq[key]=chi_square


chisq = dict(sorted(chisq.items(),key= lambda x:x[1],reverse = True))
keys,values = chisq.keys(),chisq.values()
keys = list(keys)
for i in range(20):
    print(keys[i])
