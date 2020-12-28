import tensorflow as tf
import numpy as np
import os
import sys
from collections import Counter


def find_spam_words(words):
    for word in words:
        if "biz." in word:
            return "SPAM! -" + word
        if "@trash" in word:
            return "SPAM! -" + word
        if "p://facebook" in word:
            return "SPAM! -" + word
        if "p://bbc" in word:
            return "SPAM! -" + word
        if "p://w3" in word or "w3helper." in word:
            return "SPAM! -" + word
        if "facebook." in word and not "facebook.com" in word:
            return "SPAM! -" + word
        if "@nightmail" in word or ".nightmail" in word:
            return "SPAM! -" + word
        if "@nightmail" in word:
            return "SPAM! -" + word
        if "$" in word and "000" in word:
            return "SPAM! -" + word
        if "livefilestore." in word:
            return "SPAM! -" + word


    return None

def make_dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail, encoding="Latin-1") as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    all_words += words
                    if  find_spam_words(words) != None:
                        print(find_spam_words(words))


    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if item.isalpha() == False and not str(item).endswith(".") and not str(item).endswith(",") \
                and not str(item).endswith("!") and not str(item).endswith("?"):
            #print(item)
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(3000)
    del dictionary[0:25]
    return dictionary


def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 3000))
    email_type=np.zeros((len(files)))
    docID = 0;
    fileID=0
    for file in files:
        if file.endswith(".cln"):
            email_type[fileID]=1
        elif file.endswith(".inf"):
            email_type[fileID]=0
        else:
            print("Invalid email type for training!")
            exit(-1)
        fileID = fileID + 1
        with open(file, encoding="Latin-1") as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = words.count(word)
            docID = docID + 1

    return features_matrix, email_type

def build_model(x_train,y_train):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)

if len(sys.argv) == 1:
    print("Run it with arguments!")
elif sys.argv[1] == "info":
    print("Anti_Spam_Filter_SSOSM")
    print("Haloca_Dorin")
    print("PaleVader")
    print("Version_0.1")
elif sys.argv[1] == "train":
    f_out = open(sys.argv[3], "w")

    verdict = "cln"

    dictionary = make_dictionary(sys.argv[2])
    x_train, y_train = extract_features(sys.argv[2], dictionary)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    print(x_train)
    #build_model(x_train, y_train)
    # x_test = tf.keras.utils.normalize(x_test, axis=1)

    exit(0)

    # f_read.close()
    # if verdict == "inf":
    # print(os.path.join(sys.argv[2], file + "|" + verdict))
    # f_out.write(file.replace(" ", "_") + "|" + verdict)
    # f_out.write("\n");
