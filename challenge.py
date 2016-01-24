__author__ = 'OM'

# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# sklearn
from sklearn.linear_model import LogisticRegression

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import os
import os.path

#os.chdir( "C:/Work/Challenge/word2vec-sentiments-master" )

def file_len(f):
    """Counts the number of lines in a file."""
    return sum(1 for line in f)

class DocDescription(object):
    """An auxiliary class that provides a description of the document with labelled sentences."""
    def __init__(self, size, train_size, train_label, test_label):
        self.size = size
        self.train_size = train_size
        self.test_size = self.size - self.train_size
        self.train_label = train_label
        self.test_label = test_label

class LabeledDoc(object):
    """
    Class for reading the data and splitting it into labelled sentences. Sentences for training and testing are labelled accordingly.
    'train_ratio' - sets the proportion for training and testing.
    'max_size' - sets the restriction on the maximum number of sentences in the document.

    """
    def __init__(self, source, train_label, test_label = '', train_ratio=1.0, max_size = 1000000):
        self.train_label = train_label
        self.test_label = test_label
        self.sentences = []
        try:
            with utils.smart_open(source) as fin:
                self.size = file_len(fin) if file_len(fin) < max_size else max_size
                self.train_size = int(train_ratio*self.size)
                self.test_size = self.size-self.train_size
                fin.seek(0,0)
                for item_no, line in enumerate(fin):
                    if item_no == self.size:
                        break
                    if item_no < self.train_size:
                        self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [train_label + '_%s' % item_no]))
                    else:
                        self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [test_label + '_%s' % (item_no - self.train_size)]))
        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)

    def description(self):
        return {'size': self.size, 'train_size': self.train_size, 'train_label': self.train_label, 'test_label': self.test_label}

class SentimentModel(object):
    """
    The main class that combines Doc2vec model with sentiment classification and analysis.

    """
    def __init__(self, neg_doc, pos_doc, uns_doc):
        """
        Takes in three documents that represent negative, positive and unsupervised reviews correspondingly.
        The documents should consist of labelled sentences.

        """
        self.neg_doc_desc = DocDescription(neg_doc.size, neg_doc.train_size, neg_doc.train_label, neg_doc.test_label)
        self.pos_doc_desc = DocDescription(pos_doc.size, pos_doc.train_size, pos_doc.train_label, pos_doc.test_label)
        self.sentences = neg_doc.sentences + pos_doc.sentences + uns_doc.sentences

    def get_document(self):
        return self.sentences

    def doc2vec(self, min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8):
        """Initializes Doc2vec and builds the vocabularly table."""
        self.vec_dim = size
        self.model = Doc2Vec(min_count = min_count, window = window, size = size, sample = sample, negative = negative, workers = workers)
        self.model.build_vocab(self.sentences)

    def train_doc2vec(self, rep = 10):
        """Performs training in Doc2vec model."""
        if self.model is not None:
            for t in range(rep):
                shuffle(self.sentences)
                self.model.train(self.sentences)

    def save_doc2vec(self, file_name):
        if self.model is not None:
            self.model.save(file_name)

    def load_doc2vec(self, file_name):
        self.model = Doc2Vec.load(file_name)
        self.vec_dim = self.model.vector_size

    def set_class_vecs(self):
        """Setting training and testing vectors for sentiment classification."""
        #training vectors
        train_size = self.neg_doc_desc.train_size + self.pos_doc_desc.train_size
        self.train_arrays = numpy.zeros((train_size, self.vec_dim))
        self.train_labels = numpy.zeros(train_size)

        for i in range(self.pos_doc_desc.train_size):
            prefix_train_pos = self.pos_doc_desc.train_label + '_%s' % str(i)
            self.train_arrays[i] = self.model.docvecs[prefix_train_pos]
            self.train_labels[i] = 1

        for i in range(self.neg_doc_desc.train_size):
            prefix_train_neg = self.neg_doc_desc.train_label + '_%s' % str(i)
            self.train_arrays[self.pos_doc_desc.train_size + i] = self.model.docvecs[prefix_train_neg]
            self.train_labels[self.pos_doc_desc.train_size + i] = 0

        #testing vectors
        test_size = self.neg_doc_desc.test_size + self.pos_doc_desc.test_size
        self.test_arrays = numpy.zeros((test_size, self.vec_dim))
        self.test_labels = numpy.zeros(test_size)

        for i in range(self.pos_doc_desc.test_size):
            prefix_test_pos = self.pos_doc_desc.test_label + '_%s' % str(i)
            self.test_arrays[i] = self.model.docvecs[prefix_test_pos]
            self.test_labels[i] = 1

        for i in range(self.neg_doc_desc.test_size):
            prefix_test_neg = self.neg_doc_desc.test_label + '_%s' % str(i)
            self.test_arrays[self.pos_doc_desc.test_size + i] = self.model.docvecs[prefix_test_neg]
            self.test_labels[self.pos_doc_desc.test_size + i] = 0

    def learn_model(self, classifier):
        """Training a classifier."""
        self.classifier = classifier
        self.classifier.fit(self.train_arrays, self.train_labels)

    def evaluate_model(self):
        """Evaluates the accuracy of the classification model."""
        print "The accuracy score is {:.2%}".format( self.classifier.score(self.test_arrays, self.test_labels))

def main():
    negDoc = LabeledDoc('negative_reviews.txt', 'TRAIN_NEG', 'TEST_NEG', 0.5)
    posDoc = LabeledDoc('positive_reviews.txt', 'TRAIN_POS', 'TEST_POS', 0.5)
    unsDoc = LabeledDoc('unsupervised_reviews.txt', 'TRAIN_UNS')

    sm = SentimentModel(negDoc, posDoc, unsDoc)
    sm.doc2vec()
    sm.train_doc2vec(10)
    sm.set_class_vecs()
    sm.learn_model(LogisticRegression())
    sm.evaluate_model()

if __name__ == '__main__':
    main()
