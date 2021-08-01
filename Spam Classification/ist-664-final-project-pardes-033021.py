#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Sammy Pardes
IST 664
Final Project
3/30/21
'''


# In[2]:


'''
  This program shell reads email data for the spam classification problem.
  The input to the program is the path to the Email directory "corpus" and a limit number.
  The program reads the first limit number of ham emails and the first limit number of spam.
  It creates an "emaildocs" variable with a list of emails consisting of a pair
    with the list of tokenized words from the email and the label either spam or ham.
  It prints a few example emails.
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifySPAM.py  <corpus directory path> <limit number>
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords

# define a feature definition function here

# function to read spam and ham files, train and test a classifier 
def processspamham(dirPath,limitStr):
    # convert the limit argument from a string to an int
    limit = int(limitStr)
    
    # start lists for spam and ham email texts
    hamtexts = []
    spamtexts = []
    os.chdir(dirPath)
    # process all files in directory that end in .txt up to the limit
    #    assuming that the emails are sufficiently randomized
    for file in os.listdir("./spam"):
        if (file.endswith(".txt")) and (len(spamtexts) < limit):
            # open file for reading and read entire file into a string
            f = open("./spam/"+file, 'r', encoding="latin-1")
            spamtexts.append (f.read())
            f.close()
    for file in os.listdir("./ham"):
        if (file.endswith(".txt")) and (len(hamtexts) < limit):
            # open file for reading and read entire file into a string
            f = open("./ham/"+file, 'r', encoding="latin-1")
            hamtexts.append (f.read())
            f.close()
    
    # print number emails read
    print ("Number of spam files:",len(spamtexts))
    print ("Number of ham files:",len(hamtexts))
    print
    
    # create list of mixed spam and ham email documents as (list of words, label)
    
    #make emaildocs a global variable
    global emaildocs
    
    emaildocs = []
    # add all the spam
    for spam in spamtexts:
        tokens = nltk.word_tokenize(spam)
        emaildocs.append((tokens, 'spam'))
    
    # add all the regular emails
    for ham in hamtexts:
        tokens = nltk.word_tokenize(ham)
        emaildocs.append((tokens, 'ham'))
        
    # randomize the list
    random.shuffle(emaildocs)
    
    # print a few token lists
    for email in emaildocs[:4]:
        print (email)
        print


# In[3]:


processspamham("C:/Users/slpar/OneDrive/Desktop/graduate/IST664/final-project/FinalProjectData/EmailSpamCorpora/corpus",
               1500)


# In[13]:


#get stopwords list from NLTK
stopwords = nltk.corpus.stopwords.words('english') #get basic list of stopwords

#create new list of stopwords
mystop = [":", "+", "_", "``", "'", ".", ",", "!", "/", "-", ")", "(", "*", "%", "=", "@", 
          "|", "\\", "[", "]", "?", "#", "{", "}",";", "enron","subject", "steve", "vance", 
          "susan", "lloyd", "brenda", "jackie", "howard", "stacey", "lisa", "gary", "hanks", 
          "meyers", "carlos", "donald", "julie", "taylor"]

#add nltk and custom stopwords together
all_stop = stopwords + mystop 


# In[14]:


#define function to filter out stopwords

def filter_stopwords(email_list):
    global filtered_emails #store filtered emails as a global variable
    filtered_emails = [] #initialize empty list
    for email in email_list: #for each email/category in the email in the given list (emaildocs)
        email_words = [] #initialize an empty list to store words
        for word in email[0]: #for each word in the email text
            if word.lower() not in all_stop and len(word) > 3: #if lowercased word is not a stopword and is over 3 characters 
                email_words.append(word.lower()) #append lowercased word to email_words list
        filtered_emails.append(tuple((email_words, email[1]))) #add tuple of email and category to filtered_emails list


# In[15]:


#run filter_stopwords function on emaildocs
filter_stopwords(emaildocs)

#compare filtered vs. non-filtered email
print(filtered_emails[0], '\n')
print(emaildocs[0])


# In[16]:


#extract only the words from the filtered emails
filtered_words = [] #initialize empty list
for email in filtered_emails: #for each email in the filtered_emails list
    for word in email[0]: #for each word in the email text
        filtered_words.append(word) #append the word to filtered_words

#preview some of the filtered words
print(filtered_words[:20])


# In[17]:


#get frequency distribution of filtered words
filtered_freq_dist = nltk.FreqDist(filtered_words)
filtered_freq_dist


# In[18]:


#get the top 100 most common words
filtered_freq_dist_common = filtered_freq_dist.most_common(100)

print("Most frequent filtered words with counts:", filtered_freq_dist_common[:100])


# In[19]:


#store most common words only, no counts
freq_words_only = [] #initalize empty list
for (word, count) in filtered_freq_dist_common: #for each word and count in the frequency distribution
    freq_words_only.append(word) #append only the word to freq_words_only list

#distplay top 20 most frequent words and the total length of the freq_words_only list (100)
print(freq_words_only[:20])
len(freq_words_only)


# In[20]:


#define feature function based on frequency
def freq_features(email, word_features): #initalize function given email and word_feature variables as input
    email_words = set(email) #tokenize the email, store as email_words
    features = {} #initialize empty dictionary
    for word in word_features: #for each word in the email
        features['is_freq_{}'.format(word)] = (word in email_words) #add "is_freq_", check if word is in the word_features list
    return features


# In[56]:


#run feature function on email list
freq_feature_set = [(freq_features(email, freq_words_only), category) #run freq_features() on each email given freq_words_only list and keep spam/ham classifier
                    for (email, category) in filtered_emails] #for each email and spam/ham class in filtered list

email_words = []
for email in emaildocs:
    for word in email[0]:
        email_words.append(word)

unfiltered_freq_dist = nltk.FreqDist(email_words)
unfiltered_freq_dist

unfiltered_freq_dist_common = unfiltered_freq_dist.most_common(100)

unfiltered_freq_words_only = []
for (word, count) in unfiltered_freq_dist_common:
    unfiltered_freq_words_only.append(word)

unfiltered_freq_feature_set = [(freq_features(email, unfiltered_freq_words_only), category) #run freq_features() on each email given freq_words_only list and keep spam/ham classifier
                    for (email, category) in emaildocs] #for each email and spam/ham class in filtered list

#show first email after running the freq_features function
#freq_feature_set[0]
unfiltered_freq_feature_set[0]


# In[22]:


#split data for testing and training

#get 30% of data
thirty_percent = int(len(filtered_emails)*0.3)
thirty_percent

#reserve 70% of data for training, 30% for testing
freq_train_set, freq_test_set = unfiltered_freq_feature_set[thirty_percent:], unfiltered_freq_feature_set[:thirty_percent]

#run NLTK Naive Bayes classifier on training data
freq_classifier = nltk.NaiveBayesClassifier.train(freq_train_set)

#display accuracy of running the classifier on the test data
print(nltk.classify.accuracy(freq_classifier, freq_test_set))


# In[23]:


#create confusion matrix function
def confusion_matrix(train_set, test_set, classifier):
    actual_list = [] #initalize empty lists for actual and predicted results
    predicted_list = [] 
    for (email, category) in test_set: #for each email in the test data
        actual_list.append(category) #add the true spam or ham tag to the actual_list
        predicted_list.append(classifier.classify(email)) #add the predicted class to the predicted_list
    
    #check out at the first 30 examples
    print(actual_list[:30])
    print(predicted_list[:30])

    #create a confusion matrix with ConfusionMatrix()
    cm = nltk.ConfusionMatrix(actual_list, predicted_list)
    print(cm.pretty_format(sort_by_count=True, truncate=9))
    
    #evaluation metrics 
    labels = list(set(actual_list))
    recall_list = [] #initialize empty lists
    precision_list = []
    f1_list = []
    for label in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(actual_list):
            if val == label and predicted_list[i] == label:  TP += 1
            if val == label and predicted_list[i] != label:  FN += 1
            if val != label and predicted_list[i] == label:  FP += 1
            if val != label and predicted_list[i] != label:  TN += 1
        # use these to compute recall, precision, F1
        recall = TP / (TP + FP)
        precision = TP / (TP + FN)
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    print('\tPrecision\tRecall\t\tF1')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),           "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(f1_list[i]))

confusion_matrix(freq_train_set, freq_test_set, freq_classifier)


# In[25]:


#utilize cross_validation_accuracy function
def cross_validation_accuracy(num_folds, featureset): #take number of folds, feature set as input
    subset_size = int(len(featureset)/num_folds) #create subsets depending on folds/feature set size
    print('Each fold size:', subset_size) #display subset size
    accuracy_list = [] #initalize empty accuracy_list
    
    for i in range(num_folds): #iterate over the folds
        test_this_round = featureset[(i*subset_size):][:subset_size]
        train_this_round = featureset[:(i*subset_size)] + featureset[((i+1)*subset_size):]
        #train using train_this_round
        classifier = nltk.NaiveBayesClassifier.train(train_this_round)
        #evaluate against test_this_round and save accuracy
        accuracy_this_round = nltk.classify.accuracy(classifier, test_this_round)
        print (i, accuracy_this_round)
        accuracy_list.append(accuracy_this_round)
    #find mean accuracy over all rounds
    print ('mean accuracy', sum(accuracy_list) / num_folds)

#run 10-fold validation
num_folds = 10
cross_validation_accuracy(num_folds, unfiltered_freq_feature_set)


# In[26]:


#show the most informative features
print(freq_classifier.show_most_informative_features(20))


# In[27]:


#create POS features function
def pos_features_func(email): #take email as input
    tagged_words = nltk.pos_tag(email) #run pos_tag function on the email to get parts-of-speech
    features = {} #initialize empty dictionary

    numNoun = 0 #set inital counts of parts-of-speech to 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words: #for each word and spam/ham tag in the tagged_words list
        if tag.startswith('N'): numNoun += 1 #add 1 for each POS, depending on first letter
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['20+_nouns'] = numNoun > 20 #add POS counts to dictionary, T/F depending on if total count is over 20
    features['20+_verbs'] = numVerb > 20
    features['20+_adjectives'] = numAdj > 20
    features['20+_adverbs'] = numAdverb > 20
    return features


# In[28]:


#run pos_features_func function on filtered_emails list
pos_feature_set = [(pos_features_func(email), category) 
                   for (email, category) in filtered_emails]

print(len(pos_feature_set[0][0].keys()))
print(pos_feature_set[0])


# In[29]:


#train and test the classifier
pos_train_set, pos_test_set = pos_feature_set[thirty_percent:], pos_feature_set[:thirty_percent]

pos_classifier = nltk.NaiveBayesClassifier.train(pos_train_set)

nltk.classify.accuracy(pos_classifier, pos_test_set)


# In[30]:


#perform 10-fold cross validation
num_folds = 10
cross_validation_accuracy(num_folds, pos_feature_set)


# In[31]:


confusion_matrix(pos_train_set, pos_test_set, pos_classifier)


# In[32]:


#most informative features
print(pos_classifier.show_most_informative_features(20))


# In[33]:


#create bigram features and function

#get top bigrams, save as bg_features
bigrams = list(nltk.bigrams(filtered_words))
#get frequency distribution of bigram
bg_freq_dist = nltk.FreqDist(bigrams)
#save top bigrams
bg_common = bg_freq_dist.most_common(100)

#initialize empty list
bg_features = []

#store top bigrams without counts in bg_features
for bigram in bg_common:
    bg_features.append(bigram[0])

bg_features[:20]


# In[34]:


def bigram_features(email, bigram_features): 
    email_bigrams = nltk.bigrams(email) #get bigrams for each email 
    features = {} #inialize empty dictionary
    for bigram in bigram_features: #for each bigram in the email_bigrams list
        features['bg_{}_{}'.format(bigram[0], bigram[1])] = bigram in email_bigrams #add bg_word1_word2, T/F if email has common bigrams 

    return features


# In[35]:


#run bigram_features() on filtered_emails list 
bg_feature_set = [(bigram_features(email, bg_features), category) 
                    for (email, category) in filtered_emails]

bg_feature_set[0]


# In[36]:


#train and test the classifier with 70/30 split
bg_train_set, bg_test_set = bg_feature_set[thirty_percent:], bg_feature_set[:thirty_percent]

bg_classifier = nltk.NaiveBayesClassifier.train(bg_train_set)

nltk.classify.accuracy(bg_classifier, bg_test_set)


# In[37]:


#most informative bigram features
print(bg_classifier.show_most_informative_features(20))


# In[38]:


confusion_matrix(bg_train_set, bg_test_set, bg_classifier)


# In[39]:


cross_validation_accuracy(num_folds, bg_feature_set)


# In[40]:


#combine frequency, pos, and bigram features

#take email and list of bigrams as input
def all_features(email, word_features, bigram_features):
    email_words = set(email) #tokenize email 
    email_bigrams = nltk.bigrams(email) #get email bigrams
    tagged_words = nltk.pos_tag(email) #run pos_tag function on the email to get parts-of-speech
    features = {}
    for word in word_features:
        features['is_freq_{}'.format(word)] = (word in email_words)
    for bigram in bigram_features:
        features['bg_{}_{}'.format(bigram[0], bigram[1])] = bigram in email_bigrams
    numNoun = 0 #set inital counts of parts-of-speech to 0
    numVerb = 0
    numAdj = 0
    numAdverb = 0
    for (word, tag) in tagged_words: #for each word and spam/ham tag in the tagged_words list
        if tag.startswith('N'): numNoun += 1 #add 1 for each POS, depending on first letter
        if tag.startswith('V'): numVerb += 1
        if tag.startswith('J'): numAdj += 1
        if tag.startswith('R'): numAdverb += 1
    features['20+_nouns'] = numNoun > 20 #add POS counts to dictionary, T/F depending on if total count is over 20
    features['20+_verbs'] = numVerb > 20
    features['20+_adjectives'] = numAdj > 20
    features['20+_adverbs'] = numAdverb > 20
    return features


# In[41]:


all_feature_set = [(all_features(email, freq_words_only, bg_features), category) 
                    for (email, category) in filtered_emails]

all_feature_set[0]


# In[42]:


#train and test a new classifier with 70/30 split
all_train_set, all_test_set = all_feature_set[thirty_percent:], all_feature_set[:thirty_percent]

all_classifier = nltk.NaiveBayesClassifier.train(all_train_set)

nltk.classify.accuracy(all_classifier, all_test_set)


# In[43]:


#cross-validation
cross_validation_accuracy(num_folds, all_feature_set)


# In[44]:


#most important features
print(all_classifier.show_most_informative_features(20))


# In[45]:


#confusion matrix
confusion_matrix(all_train_set, all_test_set, all_classifier)


# In[46]:


#get top trigrams
trigrams = list(nltk.trigrams(filtered_words))

tri_freq_dist = nltk.FreqDist(trigrams)

tri_common = tri_freq_dist.most_common(100)

tri_features = []

for trigram in tri_common:
    tri_features.append(trigram[0])

tri_features[:20]


# In[47]:


#define trigram features function, takes and trigram features list as input
def trigram_features(email, trigram_features):
    email_trigrams = nltk.trigrams(email)
    features = {}
    
    for trigram in trigram_features:
        features['tri_{}_{}_{}'.format(trigram[0], trigram[1], trigram[2])] = trigram in email_trigrams  
    
    return features


# In[48]:


#run trigram_features() on filtered_emails list 
tri_feature_set = [(trigram_features(email, tri_features), category) 
                    for (email, category) in filtered_emails]

tri_feature_set[0]


# In[49]:


#train and test the classifier with 70/30 split
tri_train_set, tri_test_set = tri_feature_set[thirty_percent:], tri_feature_set[:thirty_percent]

tri_classifier = nltk.NaiveBayesClassifier.train(tri_train_set)

nltk.classify.accuracy(tri_classifier, tri_test_set)


# In[50]:


#most informative bigram features
print(tri_classifier.show_most_informative_features(20))


# In[51]:


confusion_matrix(tri_train_set, tri_test_set, tri_classifier)


# In[52]:


cross_validation_accuracy(num_folds, tri_feature_set)


# In[57]:


import sys
import nltk
import random

# for testing, allow different sizes for word features
vocab_size = 100

# Function writeFeatureSets:
# takes featuresets defined in the nltk and convert them to weka input csv file
#    any feature value in the featuresets should not contain ",", "'" or " itself
#    and write the file to the outpath location
#    outpath should include the name of the csv file
def writeFeatureSets(featuresets, outpath):
    # open outpath for writing
    f = open(outpath, 'w')
    # get the feature names from the feature dictionary in the first featureset
    featurenames = featuresets[0][0].keys()
    # create the first line of the file as comma separated feature names
    #    with the word class as the last feature name
    featurenameline = ''
    for featurename in featurenames:
        # replace forbidden characters with text abbreviations
        featurename = featurename.replace(',','CM')
        featurename = featurename.replace("'","DQ")
        featurename = featurename.replace('"','QU')
        featurenameline += featurename + ','
    featurenameline += 'class'
    # write this as the first line in the csv file
    f.write(featurenameline)
    f.write('\n')
    # convert each feature set to a line in the file with comma separated feature values,
    # each feature value is converted to a string 
    #   for booleans this is the words true and false
    #   for numbers, this is the string with the number
    for featureset in featuresets:
        featureline = ''
        for key in featurenames:
            featureline += str(featureset[0][key]) + ','
        featureline += featureset[1]
        # write each feature set values to the file
        f.write(featureline)
        f.write('\n')
    f.close()

# define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'contains(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['is_freq_{}'.format(word)] = (word in document_words)
    return features

# Main program to produce movie review feature sets in order to show how to use 
#   the writeFeatureSets function
if __name__ == '__main__':
    # Make a list of command line arguments, omitting the [0] element
    # which is the script itself.
    args = sys.argv[1:]
    if not args:
        print ('usage: python save_features.py [file]')
        sys.exit(1)
    outpath = args[0]
    

    # get features sets for a document, including keyword features and category feature
    featuresets = freq_feature_set

    # write the feature sets to the csv file
    writeFeatureSets(featuresets, outpath)

    print ('Wrote spam/ham features to:', outpath)


# In[58]:


# function to read features, perform cross-validation with (several) classifiers and report results

import sys
import pandas
import numpy
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

def process(filepath):
    # number of folds for cross-validation
    kFolds = 10
    
    # read in the file with the pandas package
    train_set = pandas.read_csv(filepath)
    
    # this is a data frame for the data
    print ('Shape of feature data - num instances with num features + class label')
    print (train_set.shape)

    # convert to a numpy array for sklearn
    train_array = train_set.values

    # get the last column with the class labels into a vector y
    train_y = train_array[:,-1]
    
    # get the remaining rows and columns into the feature matrix X
    train_X = train_array[:,:-1]
    
    print('** Results from Naive Bayes')
    classifier = MultinomialNB()
    
    y_pred = cross_val_predict(classifier, train_X, train_y, cv=kFolds)
    
    # classification report compares predictions from the k fold test sets with the gold
    print(classification_report(train_y, y_pred))
    
    # confusion matrix from same
    cm = confusion_matrix(train_y, y_pred)
    #print_cm(cm, labels)
    print('\n')
    print(pandas.crosstab(train_y, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))
      

# use a main so can get feature file as a command line argument
if __name__ == '__main__':
    # Make a list of command line arguments, omitting the [0] element
    # which is the script itself.
    args = sys.argv[1:]
    if not args:
        print ('usage: python run_sklearn_model_performance.py [featurefile]')
        sys.exit(1)
    infile = args[0]
    process(infile)


# In[ ]:




