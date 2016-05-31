from __future__ import division

from numpy import *
import numpy
import matplotlib.pyplot as plt

import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
outpufile = sys.argv[1:][0]
pathToDataSet = sys.argv[2:][0]

#print outpufile
#print pathToDataSet
i=0
files = []
for arg in sys.argv:
    i=i+1
    if i>3:
        print arg
        files.append(pathToDataSet+arg)


#output.txt /home/stm/PycharmProjects/spam_classifier/data/ train-features.txt train-labels.txt test-features.txt test-labels.txt


with open(files[0], 'r') as f:
    train_features = [map(int,line.split(' ')) for line in f ]
    f.close()

with open(files[1], 'r') as f:
    train_labels = [map(int,line.split(' ')) for line in f ]
    f.close()

with open(files[2], 'r') as f:
    test_features = [ map(int,line.split(' ')) for line in f ]
    f.close()

with open(files[3], 'r') as f:
    test_labels = f.readlines()
    f.close()

print "test_labels:", len(test_labels[0])


### train set definition
train_set_list = []
train_feature_list = []
train_label_list = []
train_spam_list = []
train_nonspam_list = []

for feature, label in zip(train_features, train_labels):
    dict_train = {"train_feature": feature, "train_labels": label[0]}
    train_set_list.append(dict_train)

for i in range(0, len(train_features)):
    if train_set_list[i]["train_labels"] == 1:
        train_spam_list.append(train_set_list[i])
    elif train_set_list[i]["train_labels"] == 0:
        train_nonspam_list.append(train_set_list[i])


##################################################################
##### 4.3 ) MLE Parameters Estimation ############################
##################################################################

### number of occurences of the word j in nonspam and spam mails in the training set (Tj,0) and (Tj,1))

train_num_occur_word_in_nonspam = []  # Tj,0
train_num_occur_word_in_spam = []   # Tj,1

train_spam_list_features = []
train_nonspam_list_features = []

for item in range(0, len(train_spam_list)):
    train_spam_list_feature = train_spam_list[item]["train_feature"]
    train_spam_list_features.append(train_spam_list_feature)

for item in range(0, len(train_nonspam_list)):
    train_nonspam_list_feature = train_nonspam_list[item]["train_feature"]
    train_nonspam_list_features.append(train_nonspam_list_feature)

transpose_train_spam_list = map(list, zip(*train_spam_list_features))
transpose_train_nonspam_list = map(list, zip(*train_nonspam_list_features))

for item in range(0, len(transpose_train_spam_list)):
    spam_row_sums = sum(transpose_train_spam_list[item])
    train_num_occur_word_in_spam.append(spam_row_sums)

for item in range(0, len(transpose_train_nonspam_list)):
    nonspam_row_sums = sum(transpose_train_nonspam_list[item])
    train_num_occur_word_in_nonspam.append(nonspam_row_sums)

print "train_num_occur_word_in_nonspam:", len(train_num_occur_word_in_nonspam)
print "train_num_occur_word_in_spam:", len(train_num_occur_word_in_spam)


## number of spam emails and total number of emails in the training set
## estimate the probability that any email will be a spam

train_num_spam_emails = len(train_spam_list)
train_num_nonspam_emails = len(train_nonspam_list)
train_total_num_emails = train_num_spam_emails + train_num_nonspam_emails

prob_email_spam = float(train_num_spam_emails) / float(train_total_num_emails)
print "prob_email_spam:", prob_email_spam


### estimate the probability that a particular word in a nonspam email will be the jth word of the vocab
train_probs_nonspam_word = []
train_sum_occur_word_in_nonspam = sum(train_num_occur_word_in_nonspam)

for i in train_num_occur_word_in_nonspam:
    nonspam_prob = i / train_sum_occur_word_in_nonspam
    train_probs_nonspam_word.append(nonspam_prob)

print "train_probs_nonspam_word:", train_probs_nonspam_word

### estimate the probability that a particular word in a spam email will be the jth word of the vocab
train_probs_spam_word = []
train_sum_occur_word_in_spam = sum(train_num_occur_word_in_spam)

for i in train_num_occur_word_in_spam:
    spam_prob = i / train_sum_occur_word_in_spam
    train_probs_spam_word.append(spam_prob)

print "train_probs_spam_word:", train_probs_spam_word


### logarithms of probabilities

log_train_probs_spam_words = []
log_train_probs_nonspam_words = []

for item in train_probs_spam_word:
    if item == 0.0 :
        log_train_probs_spam_words.append(float("inf"))
    else:
        log_train_probs_spam_words.append(log(item))


print "log_train_probs_spam_words:", log_train_probs_spam_words


for item in train_probs_nonspam_word:
    if item == 0.0 :
        log_train_probs_nonspam_words.append(float("inf"))
    else:
        log_train_probs_nonspam_words.append(log(item))

print "log_train_probs_nonspam_words:", log_train_probs_nonspam_words


### multiplying test features with logarithm of train posterior probability

multiply_nonspam_test = []
nonspam_function_result = []

multiply_spam_test = []
spam_function_result = []

for row in test_features:
    multiply_spam_test.append([x*y for x,y in zip(row, log_train_probs_spam_words)  ])

for row in test_features:
    multiply_nonspam_test.append([x*y for x,y in zip(row, log_train_probs_nonspam_words)])

print "multiply_spam_test:", multiply_spam_test[0]
print "multiply_nonspam_test:", multiply_nonspam_test[0]


multiply_nonspam_test_matrix=numpy.array(multiply_nonspam_test)
s=numpy.isnan(multiply_nonspam_test_matrix)
m = numpy.isinf(multiply_nonspam_test_matrix)

multiply_nonspam_test_matrix[s]=0.0
multiply_nonspam_test_matrix[m]=0.0

sums_multiply_nonspam_test_matrix = []
for row in multiply_nonspam_test_matrix:
    sum_multiply_nonspam_test_matrix =  sum(row)
    sums_multiply_nonspam_test_matrix.append(sum_multiply_nonspam_test_matrix)

print "sums_multiply_nonspam_test_matrix:", sums_multiply_nonspam_test_matrix


multiply_spam_test_matrix=numpy.array(multiply_spam_test)
s=numpy.isnan(multiply_spam_test_matrix)
m = numpy.isinf(multiply_spam_test_matrix)

multiply_spam_test_matrix[s]=0.0
multiply_spam_test_matrix[m]=0.0

sums_multiply_spam_test_matrix = []
for row in multiply_spam_test_matrix:
    sum_multiply_spam_test_matrix =  sum(row)
    sums_multiply_spam_test_matrix.append(sum_multiply_spam_test_matrix)

print "sums_multiply_spam_test_matrix:", sums_multiply_spam_test_matrix


# Labeling test features according to MLE parameters

my_labels = []
for x,y in zip(sums_multiply_nonspam_test_matrix, sums_multiply_spam_test_matrix):
    if x > y:
        my_labels.append(0)
    else:
        my_labels.append(1)

print "MLE NB Classifier labels in test set:", my_labels
print "test_labels:", test_labels

#Predict test labels with MLA parameters

nonspam_prediction = []

for i in range(0, len(test_labels)):
    if my_labels[i] == 1:
        if my_labels[i] == int(test_labels[i]):
            nonspam_prediction.append("true")
        else:
            nonspam_prediction.append("false")

nonspam_true_predicts = nonspam_prediction.count("true")
nonspam_false_predicts = nonspam_prediction.count("false")
nonspam_accuracy = nonspam_true_predicts / (nonspam_true_predicts + nonspam_false_predicts)

print "number of MLE nonspam_true_predicts:", nonspam_true_predicts
print "number of MLE nonspam_false_predicts:", nonspam_false_predicts

print "MLE nonspam_prediction label:", nonspam_prediction
print "MLE nonspam_accuracy:", nonspam_accuracy



# ####################################################################
# # ##### 4.4 ) MAP Parameters Estimation ############################
# # ##################################################################

alpha = 1
V = 2500

map_train_num_occur_word_in_nonspam = []
for item in train_num_occur_word_in_nonspam:
    item = item + alpha
    map_train_num_occur_word_in_nonspam.append(item)

map_train_sum_occur_word_in_nonspam = sum(map_train_num_occur_word_in_nonspam)

map_train_num_occur_word_in_spam = []
for item in train_num_occur_word_in_spam:
    item = item + alpha
    map_train_num_occur_word_in_spam.append(item)

map_train_sum_occur_word_in_spam = sum(map_train_num_occur_word_in_spam)


map_train_probs_nonspam_word = []
for i in map_train_num_occur_word_in_nonspam:
    map_nonspam_prob = i / map_train_sum_occur_word_in_nonspam
    map_train_probs_nonspam_word.append(map_nonspam_prob)

print "map_train_probs_nonspam_word:", map_train_probs_nonspam_word

map_train_probs_spam_word = []
for i in map_train_num_occur_word_in_spam:
    map_spam_prob = i / map_train_sum_occur_word_in_spam
    map_train_probs_spam_word.append(map_spam_prob)

print "map_train_probs_spam_word:", map_train_probs_spam_word


### logarithms of map probabilities

map_log_train_probs_spam_words = []
map_log_train_probs_nonspam_words = []

for item in map_train_probs_spam_word:
    if item == 0.0 :
        map_log_train_probs_spam_words.append(float("inf"))
    else:
        map_log_train_probs_spam_words.append(log(item))


print "log_train_probs_spam_words:", map_log_train_probs_spam_words


for item in map_train_probs_nonspam_word:
    if item == 0.0 :
        map_log_train_probs_nonspam_words.append(float("inf"))
    else:
        map_log_train_probs_nonspam_words.append(log(item))

print "log_train_probs_nonspam_words:", map_log_train_probs_nonspam_words



### multiplying test features with logarithm of train posterior probability (MAP)

map_multiply_nonspam_test = []
map_nonspam_function_result = []

map_multiply_spam_test = []
map_spam_function_result = []

for row in test_features:
    map_multiply_spam_test.append([x*y for x,y in zip(row, map_log_train_probs_spam_words)  ])

for row in test_features:
    map_multiply_nonspam_test.append([x*y for x,y in zip(row, map_log_train_probs_nonspam_words)])

print "multiply_spam_test:", map_multiply_spam_test[0]
print "multiply_nonspam_test:", map_multiply_nonspam_test[0]


map_multiply_nonspam_test_matrix=numpy.array(map_multiply_nonspam_test)
s=numpy.isnan(map_multiply_nonspam_test_matrix)
m = numpy.isinf(map_multiply_nonspam_test_matrix)

map_multiply_nonspam_test_matrix[s]=0.0
map_multiply_nonspam_test_matrix[m]=0.0

map_sums_multiply_nonspam_test_matrix = []
for row in map_multiply_nonspam_test_matrix:
    map_sum_multiply_nonspam_test_matrix =  sum(row)
    map_sums_multiply_nonspam_test_matrix.append(map_sum_multiply_nonspam_test_matrix)

print "sums_multiply_nonspam_test_matrix:", map_sums_multiply_nonspam_test_matrix


map_multiply_spam_test_matrix=numpy.array(map_multiply_spam_test)
s=numpy.isnan(map_multiply_spam_test_matrix)
m = numpy.isinf(map_multiply_spam_test_matrix)

map_multiply_spam_test_matrix[s]=0.0
map_multiply_spam_test_matrix[m]=0.0

map_sums_multiply_spam_test_matrix = []
for row in map_multiply_spam_test_matrix:
    map_sum_multiply_spam_test_matrix =  sum(row)
    map_sums_multiply_spam_test_matrix.append(map_sum_multiply_spam_test_matrix)

print "sums_multiply_spam_test_matrix:", map_sums_multiply_spam_test_matrix


# Labeling test features according to MAP parameters

map_my_labels = []
for x,y in zip(map_sums_multiply_nonspam_test_matrix, map_sums_multiply_spam_test_matrix):
    if x > y:
        map_my_labels.append(0)
    else:
        map_my_labels.append(1)

print "MLE NB Classifier labels in test set:", map_my_labels
print "test_labels:", test_labels

#Predict test labels with MAP parameters

map_prediction = []

for i in range(0, len(test_labels)):
    if map_my_labels[i] == int(test_labels[i]):
        map_prediction.append("true")
    else:
        map_prediction.append("false")

map_true_predicts = map_prediction.count("true")
map_false_predicts = map_prediction.count("false")
map_accuracy = map_true_predicts / (map_true_predicts + map_false_predicts)

print "number of MAP email true_predicts:", map_true_predicts
print "number of MAP email_false_predicts:", map_false_predicts
print "MAP email prediction label:", map_prediction
print "MAP email accuracy:", map_accuracy


####################################################
###### 4. 5 MUTUAL INFORMATION #####################
####################################################


transpose_train_spam_list = map(list, zip(*train_spam_list_features))

scores = []
for i in range (0, len(transpose_train_nonspam_list)):
    N10 = [] # document containing term in nonspam
    for item in transpose_train_nonspam_list[i]:
        N10.append(item)

    N10_item = [] # document containing term in nonspam
    for item in N10:
        if item > 0:
            N10_item.append(item)

    num_N10 = len(N10_item) # number of document containing term in nonspam

    # print "num_s_N10:", num_N10

    N11 = []  # document containing term in spam
    for item in transpose_train_spam_list[i]:
        N11.append(item)

    N11_item = []  # document containing term in spam
    for item in N11:
        if item > 0:
            N11_item.append(item)

    num_N11 = len(N11_item) # number of document containing term in spam

    # print "num_N11:", num_N11

    N01 = []  # document not containing term in spam
    for item in N11:
        if item == 0:
            N01.append(item)

    num_NO1 = len(N01) # number of document not containing term in spam

    # print "num_NO1:", num_NO1


    NOO = [] # document not containing term in nonspam
    for item in N10:
        if item == 0:
            NOO.append(item)

    num_NO0 = len(NOO) # number of document not containing term in nonspam

    # print "num_NO1:", num_NO0

    N1_ = num_N10 + num_N11
    N0_ = num_NO0 + num_NO1
    N_1 = num_N11 + num_NO1
    N_0 = num_N10 + num_NO0

    # print "N1_:", N1_
    # print "N0_:", N0_
    # print "N_1:", N_1
    # print "N_0:", N_0

    N = num_N10 + num_NO0


    if N1_ == 0.0 or N_1 == 0.0 or N0_ == 0.0 or N_1 == 0.0 or N1_ == 0.0 or N_0 == 0.0:
        N1_ == 1 and N_1 == 1 and N0_ == 1 and N_1 == 1 and N1_ == 1 and N_0 == 1

    elif log2((N * num_N11)/ (N1_ * N_1)) ==0:
        log2((N * num_N11)/ (N1_ * N_1)) == float("inf")

    elif log2 ((N *num_NO1)/ (N0_ * N_1)) == 0:
        log2 ((N *num_NO1)/ (N0_ * N_1)) == float("inf")

    elif log2((N*num_N10)/(N1_*N_0)) == 0:
        log2((N*num_N10)/(N1_*N_0))  == float("inf")

    elif log2((N* num_NO0) / (N0_ * N_0)) ==0:
        log2((N* num_NO0) / (N0_ * N_0)) == float("inf")

    else:
        MI = (num_N11/N) * log2((N * num_N11)/ (N1_ * N_1)) +  (num_NO1/ N) * log2 ((N *num_NO1)/ (N0_ * N_1)) + (num_N10/N) * log2((N*num_N10)/(N1_*N_0)) + (num_NO0 / N) * log2((N* num_NO0) / (N0_ * N_0))


    scores.append(MI)

print "mutual_information_scores:", scores

indexes_scores = [item for item in enumerate(scores)]

print "indexed_mutual_information_scores:", indexes_scores
print "sorted, indexed_mutual_information_scores:", sorted(indexes_scores, key=lambda x: x[1], reverse=True)
print "top_ten_features with MI scores:", sorted(indexes_scores, key=lambda x: x[1], reverse=True)[0:10]


####################################################
###### 4. 6 FEATURE REDUCTION  #####################
####################################################

least_informative_list = sorted(indexes_scores, key=lambda x: x[1], reverse= True)

least_informative_feature_list = []
for item in least_informative_list:
    least_informative_list.remove(item)

#print least_informative_list[0][0]
#print least_informative_list[0][1]

# x_axis = []
# y_axis = []
# for item in least_informative_list:
#     x = least_informative_list[item]
#     #y = least_informative_list[item][1]
#
#     x_axis.append(x)
#     y_axis.append(y)
#
# print x_axis
# print y_axis





























