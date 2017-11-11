import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
import operator

#########   Part 1 - Document Similarity   ###########
def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """

    # Your Code Here
    pos_tags = nltk.pos_tag(nltk.word_tokenize(doc))
    wn_tags = [(token, convert_tag(pos)) for token, pos in pos_tags]
    synsets = [wn.synsets(token, pos=tag) for token, tag in wn_tags]
    synsets = [sets[0] for sets in synsets if len(sets) > 0]

    return synsets # Your Answer Here


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    # Your Code Here
    sim_scores = list()
    for s1_synset in s1:
        s1_synset_scores = [s1_synset.path_similarity(s2_synset) for s2_synset in s2]
        s1_synset_scores = [score for score in s1_synset_scores if score is not None]
        if len(s1_synset_scores) > 0:
            sim_scores.append(max(s1_synset_scores))
    
    return sum(sim_scores) / len(sim_scores) # Your Answer Here


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)


'''
paraphrases is a DataFrame which contains the following columns: Quality, D1, and D2.
Quality is an indicator variable which indicates if the two documents D1 and D2 are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).
'''
# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()


def most_similar_docs():
    
    # Your Code Here
    sim_scores = [(idx, document_path_similarity(row.D1, row.D2)) for idx, row in paraphrases.iterrows()]
    max_idx = max(sim_scores, key=operator.itemgetter(1))[0]

    return (paraphrases.D1[max_idx], paraphrases.D2[max_idx], sim_scores[max_idx][1]) # Your Answer Here


def label_accuracy():
    from sklearn.metrics import accuracy_score

    # Your Code Here
    sim_scores = [document_path_similarity(row.D1, row.D2) for idx, row in paraphrases.iterrows()]
    labels = [1 if score > 0.75 else 0 for score in sim_scores]

    return accuracy_score(y_pred=labels, y_true=paraphrases.Quality) # Your Answer Here


#############  Part 2 - Topic Modelling  ###############
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id_map, num_topics=10, passes=25, random_state=34)

def lda_topics():
    
    # Your Code Here
    
    return ldamodel.print_topics(num_topics=10, num_words=10)


new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]

def topic_distribution():
    
    # Your Code Here
    new_doc_transformed = vect.transform(new_doc)
    corpus = gensim.matutils.Sparse2Corpus(new_doc_transformed, documents_columns=False)
    doc_topics = ldamodel.get_document_topics(corpus)
    
    return list(doc_topics)[0] # Your Answer Here


def topic_names():
    
    return ["Education", "Automobiles", "Computers & IT", "Religion", "Automobiles", "Sports", "Health", "Religion", "Computers & IT", "Science"] # Your Answer Here