import glob
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import config

config = config.BaseConfig().get_args()

# load test set
df_test = pd.read_csv(config.test_set, header=None, sep=';',
                      quotechar="'", names=['label', 'text'])
filenames = sorted(glob.glob(config.train_set))

for filename in filenames:  # for each subset

    df_train = pd.read_csv(filename, header=None, sep=';', quotechar="'", names=['label', 'text'])

    # build the classifier pipeline
    lsvc_classifier = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer(
            sublinear_tf=True  # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
        )),
        ('clf', LinearSVC(
            dual=False,
            C=1.6,
            class_weight="balanced"
        ))
    ])

    lsvc_classifier.fit(df_train['text'], df_train['label'])  # train the classifier
    predicted = lsvc_classifier.predict(df_test['text'])  # predict the test set
    acc = np.mean(predicted == df_test['label'])  # calculate the accuracy
    print(acc)
    # precision = precision_score(df_test['label'],predicted)
    # f1_score = f1_score(df_test['label'], predicted)
    # print("f1_score", f1_score)
    print(filename[16:-4], "%.2f" % float((100 - acc * 100)), sep=" -> ")  # print the error rate
    cm = confusion_matrix(df_test['label'], predicted)
    print(cm)
    cr = classification_report(df_test['label'], predicted)
    print(cr)


def get_label(input_path):
    input_doc = pd.read_csv(input_path, header=None, sep=';',
                      quotechar="'", names=['label', 'text'])
    predictions = lsvc_classifier.predict(input_doc['text'])
    file1 = open("../result.txt", "w")
    file1.writelines(predictions)
    file1.close()
    print("You can access the result on result.txt file")
    return predictions.tolist()[1]

get_label(config.test_set)