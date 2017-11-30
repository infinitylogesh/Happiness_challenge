import spacy

nlp = spacy.load('en')

# Library to pre-process text - lemmatize,stop_words removal etc.
# spacy is used for all the tasks.

class process(object):

    doc = None
    value = None
    tokens = None

    #  when the class is initialized , Doc is created for the given sentence.
    def __init__(self,sentence):
        self.doc = self.get_spacy_doc(sentence)
        self.value = sentence
        self.tokens = [token.text for token in self.doc]

    def get_spacy_doc(self,sentence):
        return nlp(unicode(sentence))

    # @property is included to remove the paranthesis while calling this function
    # function returns the instance of the class for chaining.
    @property
    def lemmatize(self):
        self.value = ' '.join([word.lemma_ for word in self.doc])
        return self.value

    #  Stop words are removed and the value is saved in the self.value variable.
    @property
    def remove_stop_words(self):
        self.value = ' '.join([word.text for word in self.doc if word.is_stop is False])
        return self.value

    def pos_tag(self):
        self.value = ' '.join([word.pos for word in self.doc])
        return self.value

