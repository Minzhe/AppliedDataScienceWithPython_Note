import nltk
import pandas as pd
import numpy as np
import operator
import re
from nltk.corpus import words

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()
    
# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)


def answer_one():
    
    return round(len(set(text1))/len(text1),3) # Your answer here

answer_one()

def answer_two():
    
    whale = [w for w in text1 if re.search(r'^[Ww]hale$',w)] 
    return (len(whale)/len(text1)) * 100   # Your answer here

answer_two()

def answer_three():
    
    fdist = nltk.FreqDist(text1)
    token_sorted = sorted(fdist.items(), key=operator.itemgetter(1), reverse=True)
    return token_sorted[:20]

answer_three()

def answer_four():
    
    fdist = nltk.FreqDist(text1)
    words = fdist.keys()
    words = [w for w in words if len(w) > 5 and fdist[w] > 150]
    return sorted(words) # Your answer here

answer_four()

def answer_five():
    
    words_length = [len(w) for w in text1]
    max_idx = words_length.index(max(words_length))
    return (text1[max_idx], words_length[max_idx])

answer_five()

def answer_six():
    
    fdist = nltk.FreqDist(text1)
    fdict_filter = [(fdist[key], key) for key in fdist.keys() if key.isalpha() and fdist[key] > 2000]
    fdict_filter = sorted(fdict_filter, key=lambda x: x[0], reverse=True)
    return fdict_filter

answer_six()

def answer_seven():
    
    moby_sent = nltk.sent_tokenize(moby_raw)
    moby_token = nltk.word_tokenize(moby_raw)
    return len(moby_token) / len(moby_sent)

answer_seven()

def answer_eight():
    
    pos = nltk.pos_tag(text1)
    tag_fdist = nltk.FreqDist(tag for (word, tag) in pos)
    tag_common = tag_fdist.most_common()[:5]
    return tag_common

correct_spellings = words.words()

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):
    
    rcmd = []
    for entry in entries:
        spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
        dist_list = [nltk.jaccard_distance(set(nltk.ngrams(entry, n=3)), set(nltk.ngrams(spell, n=3))) for spell in spell_list]
        min_idx = dist_list.index(min(dist_list))
        rcmd.append(spell_list[min_idx])
    return rcmd
    
answer_nine()

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):
    
    rcmd = []
    for entry in entries:
        spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
        dist_list = [nltk.jaccard_distance(set(nltk.ngrams(entry, n=4)), set(nltk.ngrams(spell, n=4))) for spell in spell_list]
        min_idx = dist_list.index(min(dist_list))
        rcmd.append(spell_list[min_idx])
    return rcmd
    
answer_ten()

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):
    
    rcmd = []
    for entry in entries:
        spell_list = [spell for spell in correct_spellings if spell.startswith(entry[0]) and len(spell) > 2]
        dist_list = [nltk.edit_distance(entry, spell, transpositions=True) for spell in spell_list]
        min_idx = dist_list.index(min(dist_list))
        rcmd.append(spell_list[min_idx])
    return rcmd 
    
answer_eleven()