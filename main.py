import tensorflow as tf
from urllib.parse import urlparse
import numpy as np
import os
import sys
from collections import Counter
import re

DICTIONARY_SIZE = 10000


def find_spam_words(words):
    for word in words:
        if "biz." in word:
            return "SPAM! -" + word
        if "@trash" in word:
            return "SPAM! -" + word

            return "SPAM! -" + word

    return None


def process_words(words):
    for index, item in enumerate(words):
        words[index] = words[index].lower()

        if item.isalpha() == False:
            print("NON-ALPHA " + words[index])
            if words[index].isalpha() == False:
                can_be_link = False
                try:
                    if 'w.' in words[index] or "://" in words[index]:
                        can_be_link = True
                    if can_be_link == True:
                        re.search(r"(h?t?t?p?s?://)?(w{0,3}\.)?[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]

                except:
                    #print("Not link!! " + words[index])
                    can_be_link = False

                if str(words[index]).startswith('$') and len(words[index]) >=1+4:
                    words[index] = "$$$"
                    print(words[index])
                elif str(words[index]).startswith('usd$') and len(words[index]) >=4+4:
                    words[index] = "usd$$$"
                    print(words[index])
                    new_word = list()
                    new_word.append("$$$")
                    words = words + new_word
                    print(words)
                elif str(words[index]).startswith('$') and len(words[index]) >= 1 + 2:
                    words[index] = "$$"
                    print(words[index])
                    print(words)
                elif str(words[index]).startswith('$') and len(words[index]) >= 1 + 1:
                    words[index] = "$"
                    print(words[index])
                    print(words)
                elif '@' in words[index] and re.search(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", words[index]):
                    words[index] = re.search(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]
                    print(">CONTINE email " + words[index])
                    new_word=list('')
                    new_word.append(re.search(r"@[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0][1:])
                    words = words + new_word
                    print(words)

                elif can_be_link == True :
                    print(">CONTINE link " + words[index])
                    words[index] = re.search(r"(h?t?t?p?s?://)?(w{0,3}\.)?[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]


                    print(">CONTINE link1 " + words[index])
                    if urlparse(words[index]).netloc:
                        domain = urlparse(words[index])
                        if '.' in domain.netloc:
                            if domain.netloc.startswith('w') and 'w.' in str(domain.netloc):
                                words[index] = domain.scheme + "://" + domain.netloc.split('.')[0] + "."+domain.netloc.split('.')[-2] + "." + \
                                               domain.netloc.split('.')[-1]
                            else:
                                words[index] = domain.scheme + "://" + domain.netloc.split('.')[-2] + "." + \
                                               domain.netloc.split('.')[-1]

                            print(">CONTINE link2 " + domain.netloc.split('.')[-2] + "." + domain.netloc.split('.')[-1])
                            # adaug domeniul
                            new_word = list()
                            new_word.append(domain.netloc.split('.')[-2] + "." + domain.netloc.split('.')[-1])
                            words = words + new_word

                            # adaug domeniul tarii
                            new_word = list()
                            new_word.append("." + domain.netloc.split('.')[-1])
                            words = words + new_word

                            #adaug http
                            new_word = list()
                            new_word.append(domain.scheme)
                            words = words + new_word
                        else:
                            words[index] = domain.netloc

                        print(">CONTINE link3 " + words[index])

                    else:
                        new_word = list()
                        new_word.append("no_http")
                        words = words + new_word
                        print("NO HTTP!" + words[index])
                        if words[index].startswith('w') and 'w.' in words[index]:
                            words[index] = words[index].split('.')[0] + "." + \
                                           words[index].split('.')[-2] + "." + \
                                           words[index].split('.')[-1]
                        else:
                            words[index] = words[index].split('.')[-2] + "." + words[index].split('.')[-1]

                        new_word = list()
                        new_word.append(words[index].split('.')[-2] + "." + words[index].split('.')[-1])
                        words = words + new_word
                    print(words)
                elif str(words[index]).startswith('<') and False:
                    print(">GASIT tag " + item)
                    words[index] = ' '
                else:
                    if str(words[index]).endswith('.'):
                        words[index] = words[index].replace('.', '')
                    if '\'' in words[index]:
                        words[index] = words[index].replace('\'', '')
                    if '\"' in words[index]:
                        words[index] = words[index].replace('\"', '')
                    if str(words[index]).endswith(','):
                        words[index] = words[index].replace(',', '')
                    if str(words[index]).endswith('!'):
                        words[index] = words[index].replace('!', '')
                    if str(words[index]).endswith('?'):
                        words[index] = words[index].replace('?', '')
                    if str(words[index]).endswith(':'):
                        words[index] = words[index].replace(':', '')
                    if str(words[index]).endswith(';'):
                        words[index] = words[index].replace(';', '')
                    if str(words[index]).startswith('['):
                        words[index] = words[index].replace('[', '')
                    if str(words[index]).endswith(']'):
                        words[index] = words[index].replace(']', '')
                    if str(words[index]).startswith('('):
                        words[index] = words[index].replace('(', '')
                    if str(words[index]).endswith(')'):
                        words[index] = words[index].replace(')', '')
                    words[index] = words[index].replace('-', ' ')
                    if words[index].isalpha() == False:
                        print(">NON-ALPHA " + words[index])
                        words[index] = ' '
    return words


def make_dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail, encoding="Latin-1") as m:
            for i, line in enumerate(m):
                if i == 2:  # Body of email is only 3rd line of text file
                    words = line.split()
                    words = process_words(words)  # parse URL, remove non-alpha words
                    all_words += words
                    # if  find_spam_words(words) != None:
                    #    print(find_spam_words(words))

    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if len(item) == 1:
            del dictionary[item]

    dictionary = dictionary.most_common(DICTIONARY_SIZE)
    # del dictionary[0:25]
    # print(dictionary[0:DICTIONARY_SIZE])
    return dictionary


def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), DICTIONARY_SIZE))
    email_type = np.zeros((len(files)))
    docID = 0
    fileID = 0
    for file in files:
        if file.endswith(".cln"):
            email_type[fileID] = 0
        elif file.endswith(".inf"):
            email_type[fileID] = 1
        else:
            print("Invalid email type for training!")
            exit(-1)
        fileID = fileID + 1
        with open(file, encoding="Latin-1") as fi:
            for i, line in enumerate(fi):
                if i == 2:
                    words = line.split()
                    process_words(words)
                    for word in words:
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[docID, wordID] = features_matrix[docID, wordID] + words.count(word)
            docID = docID + 1

    return features_matrix, email_type


def build_model(x_train, y_train):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=8)
    return model


def classify_emails_train(mail_dir, model, x_test, output_file):
    f = open(output_file, "w")
    predictions = model.predict([x_test])

    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), DICTIONARY_SIZE))
    email_type = np.zeros((len(files)))
    docID = 0
    fileID = 0
    for i, file in enumerate(files):

        if predictions[i] >= 0.5:
            f.write(file + "|inf")
        elif predictions[i] < 0.5:
            f.write(file + "|cln")
        else:
            f.write("Classification error!")
        f.write("\n")
    f.close()


if len(sys.argv) == 1:
    print("Run it with arguments!")
elif sys.argv[1] == "info":
    print("Anti_Spam_Filter_SSOSM")
    print("Haloca_Dorin")
    print("PaleVader")
    print("Version_0.1")
elif sys.argv[1] == "train":
    dictionary = make_dictionary(sys.argv[2])
    x_train, y_train = extract_features(sys.argv[2], dictionary)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    #model = build_model(x_train, y_train)
    #model.save("spam_filter.model")

    f = open("dictionary.txt", "w")
    f.write(str(dictionary[0:DICTIONARY_SIZE]))
    f.close()

elif sys.argv[1] == "train+scan":
    verdict = "cln"

    dictionary = make_dictionary(sys.argv[2])
    x_train, y_train = extract_features(sys.argv[2], dictionary)

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    model = build_model(x_train, y_train)

    x_test, y_test = extract_features(sys.argv[3], dictionary)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    classify_emails_train(sys.argv[3], model, x_test, sys.argv[4])
    # print("loss: " + str(val_loss) + " - acc: " + str(val_acc))

    f = open("dictionary.txt", "w")
    f.write(str(dictionary[0:DICTIONARY_SIZE]))
    f.close()



    # f_read.close()
    # if verdict == "inf":
    # print(os.path.join(sys.argv[2], file + "|" + verdict))
    # f_out.write(file.replace(" ", "_") + "|" + verdict)
    # f_out.write("\n");
