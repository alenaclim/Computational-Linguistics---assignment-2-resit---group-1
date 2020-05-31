#!/usr/bin/env python
# coding: utf-8

# In[ ]:


Distributional Semantic Space


# For this assignment I implemented a word2vec technique, and to understand this I used the following resources:
# 
# -the lectures for Dimensional Space and for Word2Vec, to understand the theory behind it;
# 
# -these 3 tutorials about implementing word2vec in python:
#     https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
#     https://stackabuse.com/implementing-word2vec-with-gensim-library-in-python/
#     https://github.com/buomsoo-kim/Word-embedding-with-Python/blob/master/word2vec/source%20code/word2vec.ipynb
# 
# After going through them first, step by step, with this corpus, and urderstanding very good how to do it, 
# I took your code apart to understand each section, and then managed to put the output of the word2vec model into a 2dnumpy.
# 
# I didn't copy the code provided in these tutorials, but I used the technique explained by them, with the 
# gensim library. I hope that's alright.
# 
# Regarding the rho, I beat the baseline immediately after implementing word2vec instead of the cooccurrence 
# technique (that you provided). But I tried to make it between .6 and .8, as you mentioned that would could as a "good"
# rho in general. By increasing the window size to 10, and the iterator to 15 or 20, I managed to do that on the dev set. 
# Fingers crossed on the test set, no matter what I did, on this corpus I couldn't explain more than 1922 words :(
# I think I need to try other technique to get a higher rho than .627.
# 
# Parameters: t=0 (frequecy), size = 100, window = 10, min-count = 1 (taking all words), iter = 20.
# 
# Also, for the next part, I'll ask you where did you manage to download your corpus from. I tried so many websites, but 
# I think even if I manage to download them, I cannot understand the files. But that's going to come next.
# 

# In[2]:


import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import spearmanr
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


# To implement the word2vec method, I used the gensim library
#pip install --upgrade gensim

from gensim.models import Word2Vec


# In[4]:


with open('/_MYstuff/Desktop/Uni/Computational Linguistics/corpus.json') as f:
    data = json.load(f)
    
#data


# In[5]:


norms = pd.read_csv('/_MYstuff/Desktop/Uni/Computational Linguistics/norms.dev.csv', header = 0, sep = ' ')


# In[20]:


# In this class I added one more method, that's calculating the word2vec model, and 
# I changed the self.embeddings to the 2dnumpy array I created instead of the cooccurences

class SemanticSpace(object):
    
    """
    This class creates distributional semantic space using a co-occurrence based approach.
    
    Any update you plan on doing (applying weighting schemes or dimensionality reduction, use a
    prediction-based DSM, ...) need to be implemented in this class.
    
    In its current implementation, this class gets the corpus as a list of lists, with inner lists 
    consisting of strings. You can change anything you want: use different corpora, in different 
    formats, apply feature weighting schemes to the raw co-occurrences, apply dimensionality reduction,
    learn the embeddings using prediction-based methods, ...
    
    The important things are:
    - the SemanticSpace class needs to have the  word2idx attribute (a dictionary mapping strings to 
        integers, representing the row indices in the embedding space corresponding to each word). The 
        keys of this dictionary must be lower-cased tokens!
    - the SemanticSpace class needs to have the embeddings attribute (a 2d NumPy array where rows are 
        words and columns are dimensions, symbolic or latent)
    """
    
    def __init__(self, corpus, t=10):
        
        """
        To initialize the semantic space it is enough to provide a corpus as a list of lists,
        where each inner list consists of strings, representing the words in a sentence, and to
        indicate a frequency threshold: only words with a frequency count higher or equal to the
        threshold are used to build the semantic space.
        """
        
        self.corpus = corpus
        
        # compute a word frequency distribution
        self.freqs = self.freq_distr()
        
        # select words which occur more often than the threshold
        self.targets = {w for w, f in self.freqs.items() if f > t}
        
        # map words to numerical indices
        self.word2idx = {w: i for i, w in enumerate(self.targets)}
        
        # update a co-occurrence matrix.
        self.embeddings = self.word_2_vec()
        
        """
        IMPORTANT: the actual semantic space needs to be encoded as a NumPy 2d array!
        Whatever transformation you decide to apply to raw counts (or whether you want to use Word2Vec)
        make sure that self.embeddings is a 2d NumPy array with as many rows as there are words in the 
        vocabulary. I will compute cosine similarity indexing the embedding space!
        """
    def word_2_vec(self):
        # This far this was the best selection of the parameters
        model = Word2Vec(self.corpus, size=100, sg = 1, window = 10, min_count = 1, iter = 20) 
        
        rows = len(self.targets)
        
        vector_space = np.zeros([rows,100])
        
        for sentence in self.corpus:
            for word in sentence:
                if word in self.targets:
                    vector_space[self.word2idx[word]] = model.wv[word]
        return vector_space
    
    def freq_distr(self):
        
        word_frequencies = Counter()
        for sentence in self.corpus:
            for word in sentence:
                word_frequencies[word] += 1
        return word_frequencies
    
    def harvest_counts(self):
        
        # initialize an empty 2d NumPy array
        cooccurrences = np.zeros((len(self.targets), len(self.targets)))
        checkpoints = {int(len(self.corpus)*perc): perc*100 for perc in [0.2, 0.4, 0.6, 0.8, 1]}

        for i, sentence in enumerate(self.corpus):
            for word in sentence:
                for context in sentence:
                    # consider a whole sentence as context and update counts only between target words
                    if word in self.targets and context in self.targets and word != context:
                        cooccurrences[self.word2idx[word], self.word2idx[context]] += 1

            # it's a long computation, let's make sure that we're making progress
            if i+1 in checkpoints:
                print("{}% of the corpus sentences have been processed at {}".format(
                    checkpoints[i+1], datetime.utcnow())
                     )

        return cooccurrences


# In[12]:


# I did not change anything in this class

class Sim(object):
    
    """This class compares semantic similarity scores retrieved from a corpus to human-generated norms."""
    
    def __init__(self, norms, semantic_space):
        
        """
        This class is initialized providing two input structures:
        - a Pandas DataFrame: the first column ('w1') contains the first word in the similarity pair, 
            the second column ('w2') contains the second word in the similarity pair, the third column 
            ('sim') contains the similarity score between w1 and w2.
        - an object of class SemanticSpace (check the docs of this class for what it consists of and what 
            the necessary attributes it needs to have are)
        
        Don't change this class at all! Make sure that it works with the SemanticSpace class as you modify it.
        I will use this class to evaluate your submissions, specifically the compute_correlation() method.
        """
        
        self.norms = norms
        self.embeddings = semantic_space.embeddings
        self.word2idx = semantic_space.word2idx
        
    def compute_similarity(self, w1, w2):
        
        try:
            e1 = self.embeddings[self.word2idx[w1], :]
            try:
                e2 = self.embeddings[self.word2idx[w2], :]
                s = cosine_similarity(e1.reshape(1, -1), e2.reshape(1, -1))
                return s[0][0]
            
            except KeyError:
                print("Couldn't find the embedding for word {}. Not computing cosine for pair {}-{}.".format(
                    w2, w1, w2)
                     )
                
        except KeyError:
            print("Couldn't find the embedding for word {}. Not computing cosine for pair {}-{}.".format(
                    w1, w1, w2)
                     )
        
        return None
    
    def compute_correlation(self):
        
        true_similarities = []
        estimated_similarities = []
        for _, row in self.norms.iterrows():
            s = self.compute_similarity(row['w1'], row['w2'])
            if s:
                estimated_similarities.append(s)
                true_similarities.append(row['sim'])
        
        print("Pairs for which it was possible to compute cosine similarity: {}".format(
            len(estimated_similarities))
             )
        
        print("Spearman rho between estimated and true similarity scores: {}".format(
            spearmanr(true_similarities, estimated_similarities)[0])
             )
        
        


# In[21]:


S = SemanticSpace(data[0], t=0)


# In[9]:


# just checking that my idexes are correct (I had problems with the indexing in the beginning)

#S.word2idx


# In[22]:


sim = Sim(norms, S)


# In[23]:


sim.compute_correlation()


# In[ ]:





# In[ ]:


# Alena's Distributional Semantic Space (using word 2 vec): changing differenct parameters, and deciding on the best

# t=0, size = 100, the rest were default, it explained 1922, rho=0.46
# t=0, size = 100, window = 3, sg = 1, window = 3, min_count = 1, iter = 10, explained 1922, rho=0.56
# t=0, size = 100, window = 5, ..., explained 1922, rho=0.595
# t=10, size = 100, ..., explained 1660, rho=0.628 # not changing the frequency again haha
# t=0, size = 100, window = 7, ..., explained 1992, rho = 0.602 #0.5973027394084918
# t=0, size = 100, window = 10, ..., explained 1992, rho = 0.6145 # 0.6110385663024789 #0.617686246938297
# size=150, window =10, 1922, 0.605426884330648
# size-200, window = 10, 1922, 0.6030114262194551
# size=100, window =10, sg=1, iter to 15 => 0.6245213133151273

# size=100, window =10, sg=1, iter to 20 => 0.6273466998188015 -> THIS IS IT :D

# size=300, window =10, sg=1, inter=20 => 1992 explained, rho = 0.6120727399179045 => i'll keep the previous score


# In[ ]:


# Giovanni's Distributional Semantic Space (using a cooccurrence technique)

# t=0: 965 pairs in the test, 49K tokens in the embedding space, rho=0.2239
# t=10: 829 pairs in the test, 12K tokens in the embedding space
# t=25: 714 pairs in the test, 8K tokens in the embedding space


# In[ ]:





# In[ ]:




