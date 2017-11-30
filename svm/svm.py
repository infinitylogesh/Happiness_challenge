#Import Library
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import sys
from subprocess import Popen
from pre_process import process
from sklearn.externals import joblib
import numpy as np
import pickle

reload(sys)
sys.setdefaultencoding('utf8')

# Create SVM classification object
dataset_file = "/Volumes/D/Design/garage/ml/datasets/hackerearth_sentiment/train.csv"
raw_dataset = pd.read_csv(dataset_file)
classes = raw_dataset["Is_Response"].drop_duplicates().values.tolist()
le = LabelEncoder()
le.fit(classes)
class_with_index = le.classes_.tolist()

# doc2vec = pretrained_doc2vec(compressed_doc2vec_model)
y = raw_dataset["Is_Response"]
print class_with_index

trainset,testset = train_test_split(raw_dataset,test_size=0.25,random_state=42)

#Creating training test and validation sets
trainset_pre_processed = [process(sentence).lemmatize for sentence in trainset["Description"].values]
testset_pre_processed = [process(sentence).lemmatize for sentence in testset["Description"].values]

stop_words = ['in','of','is','and','for']

#Removing the stop word feature in the tfidf
tfidf = TfidfVectorizer(ngram_range=(1,3),stop_words=stop_words)
#Applying tfidf model for training
tfidf.fit(trainset_pre_processed+testset_pre_processed)

print "**** TFIDF fit is completed ****"

train_data_features = tfidf.transform(trainset_pre_processed)
y_train = le.transform(trainset["Is_Response"].values)

test_data_features = tfidf.transform(testset_pre_processed)
y_test = le.transform(testset["Is_Response"].values)

print "*** Train data and test data are formed ***"

#print type(train_data_features)
#print type(y_train)
# #Returns the array using transform
# train_data_features = tfidf.transform(x_train)
# test_data_features = tfidf.transform(x_test)
# validate_data_features = tfidf.transform(x_validate)

#using SVM model for training the dataset and balancing the weightage of all the words present

model = svm.SVC(C=10.0, cache_size=200, class_weight='balanced', coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear',
                max_iter=-1, probability=True, random_state=42, shrinking=True,
                tol=0.001, verbose=False)


model.fit(train_data_features, y_train)

print "*** Model training is done ***"

# model = joblib.load('svm_intent_classification_v01_accuracy1.pkl')

#Accuracy score for test set

score = model.score(test_data_features, y_test)
print "Test Score: "+ str(score)

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
