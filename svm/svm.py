#Import Library
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import sys,os,datetime
import argparse
from subprocess import Popen
from pre_process import process
from sklearn.externals import joblib
import numpy as np
import pickle

reload(sys)
sys.setdefaultencoding('utf8')

class model(object):

    def __init__(self):
        self.top_nwords = 500
        self.dataset_file = "../dataset/train.csv"
        self.raw_dataset = pd.read_csv(self.dataset_file)
        self.classes = self.raw_dataset["Is_Response"].drop_duplicates().values.tolist()
        self.le = LabelEncoder()
        self.le.fit(self.classes)
        self.class_with_index = self.le.classes_.tolist()
        self.y = self.raw_dataset["Is_Response"]
        self.stop_words = ['in','of','is','and','for']
        self.orchestrate()

    def orchestrate(self):
        self.train_test_split()
        self.pre_process_text()
        self.vectorize()
        self.model_train()
        score = self.model_test()
        self.write_results(score)

    def train_test_split(self):
        self.trainset,self.testset = train_test_split(self.raw_dataset,test_size=0.25,random_state=42)
        #self.trainset = self.trainset[:10]

    def pre_process_text(self):
        #Creating training test and validation sets
        self.trainset_pre_processed = [process(sentence).lemmatize for sentence in self.trainset["Description"].values]
        self.testset_pre_processed = [process(sentence).lemmatize for sentence in self.testset["Description"].values]

    def vectorize(self):
        #Removing the stop word feature in the tfidf
        self.tfidf = TfidfVectorizer(ngram_range=(1,3),stop_words="english",max_features=self.top_nwords)
        #Applying tfidf model for training
        self.tfidf.fit(self.trainset_pre_processed+self.testset_pre_processed)
        print self.get_timestring+ ": **** TFIDF fit is completed ****"
        self.train_data_features = self.tfidf.transform(self.trainset_pre_processed)
        self.y_train = self.le.transform(self.trainset["Is_Response"].values)
        self.test_data_features = self.tfidf.transform(self.testset_pre_processed)
        self.y_test = self.le.transform(self.testset["Is_Response"].values)
        print self.get_timestring + ": *** Train data and test data are formed ***"

    def model_train(self):
        print self.get_timestring + ": *** Model training has started ***"
        self.model = svm.SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                    max_iter=-1, probability=True, random_state=42, shrinking=True,
                    tol=0.001, verbose=False)
        self.model.fit(self.train_data_features, self.y_train)
        joblib.dump(self.model,"svm_happiness_challenge_"+self.get_timestring+".pkl")
        print self.get_timestring + ": *** Model training is done ***"

    def load_pickled_model(self):
        self.model = joblib.load('svm_intent_classification_v01_accuracy1.pkl')

    def model_test(self):
        #Accuracy score for test set
        score = self.model.score(self.test_data_features, self.y_test)
        return score

    def write_results(self,score):
        text =  "Test Score: "+ str(score) + "\n"
        text += str(self.model) + "\n"
        with open("results_"+self.get_timestring+".txt",'w+') as f:
            f.write(text)
    @property
    def get_timestring(self):
        return str(datetime.datetime.now()).split('.')[0].replace(' ','_')


#Predict Output
#predicted = model.predict(test_data_features)
#print predicted[:40]
#print y_test[:40]
#print x_test[:40]
#print testset[0].values[:40]
#log_predictility = model.predict_log_proba(validate_data_features[0])
#print log_predictility,class_with_index[np.argmax(log_predictility,axis=1)]
#validation_predict = model.predict(validate_data_features)
# print validation_predict
# print type(validation_predict)
#Validation for fact based query
#score_validation = model.score(validate_data_features, y_validate)
#print "validation score",score_validation
#Validation for events-list query
#Prediction(.predict) is only required if needed to put display the predictions
#events_validation_predict = model.predict(events_validate_data_features)
#events_score_validation = model.score(events_validate_data_features, y_events_validate)
#print "events_validation score",events_score_validation


#Predicting and storing the validation set results
"""
print "Do you want to commit? Y(1) / N(0)"
git_commit_flag = int(raw_input())
if (git_commit_flag == 1):
    print "Enter the features u changed"
    print "cvalue, dataset_training_v,stopwords, lemmatization, validation_set"
    feature_change = raw_input()

#Kept out of if block as it requires to update the results file even if it is not commited
target = open('results.txt', 'w+')
target.write("test set accuracy:")
# target.write(str(score))
target.write("\n")
target.write("validation set accuracy:")
target.write(str(score_validation))
target.write("\n")
target.write("\n")

print "Do you want to append the contents to the two files? Y(1) / N(0)?"
prediction_file = int(raw_input())
wrong_predicted_file = None
if (prediction_file == 1):
    wrong_predicted_file = open('wp_file_svm.txt', 'a+')

for index, sentence_vector in enumerate(events_validate_data_features):
    prediction_intent = class_with_index[int(events_validation_predict[index])]
    original_intent = class_with_index[int(y_events_validate[index])]


    if(prediction_intent != original_intent and prediction_file == 1):

        wp_target = raw_events_validation[0][index]+",\t"+prediction_intent+",\t"+original_intent+",\t"+feature_change+",\t"+str(events_score_validation)+"\n"
        #wp_target = sentence_vector +",\t"+ str(prediction_intent)+",\t"+str(original_intent)
        wrong_predicted_file.write(wp_target)


    target.write(raw_events_validation[0][index])
    target.write(", ")
    target.write(prediction_intent)
    target.write(", ")
    target.write(original_intent)
    target.write("\n")


if prediction_file == 1:
    wrong_predicted_file.close()

target.close()
# if (git_commit_flag == 1):
#     Popen(["git","add","results.txt"],
#           cwd = '../intent_classification/model_prediction')
#     commitMessage = "Accuracy "+str(score_validation)+"message "+feature_change
#     Popen(["git","commit","results.txt","-m",commitMessage],
#           cwd = '../intent_classification/model_prediction')

#Saving the model
joblib.dump(model, './model/svm_intent_classification_v05_accuracy.pkl')


#Custom validation inputs and getting the prediction
while(True):
    print "Enter the query for prediction"
    query = raw_input(">>")
    query = process(query).lemmatize
    vectorized = tfidf.transform([query])

    print class_with_index[int(model.predict(vectorized))]

"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--remote',type=int,default=0,help="""To control if the program run on papaerspace or in local""")
    FLAGS, unparsed = parser.parse_known_args()
    mdl = model()
    if FLAGS.remote==1:
        os.system("halt")

