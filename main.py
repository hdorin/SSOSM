import tensorflow as tf
from urllib.parse import urlparse
import numpy as np
import os
import sys
from collections import Counter
import re
import json

DICTIONARY_SIZE = 6000


def is_time(word):
    try:
        time = re.search(r"[0-9]{1,2}(:| )[0-9]{1,2}((:| )[0-9]{1,2})?", word)[0]
    except:
        return False

    return time

def is_date(word):
    try:
        date = re.search(r"[0-9]{1,4}(/| )[0-9]{1,4}(/| )[0-9]{1,4}", word)[0]
    except:
        return False

    return date

def is_year(word):
    try:
        year = re.search(r"(19[0-9]{2})|(20[0-9]{2})", word)[0]
    except:
        return False

    return year

def is_normal_word(word):
    #try:
    #    re.search(r"[a-z0-9></]", word)[0]
    #except:
    #    return False

    nr=0
    for item in word:
        if item in "abcdefghijklmnopqrstuvwxyz":
            nr = nr + 1
        elif item in "0123456789":
            nr = nr + 1
    if nr == 0 or len(word)/nr > 4:
        return False


    return True

def process_words(words):
    for index, item in enumerate(words):
        words[index] = words[index].lower()

        can_be_link = False
        can_be_email = False

        try:
            if 'w.' in words[index] or "://" in words[index]:
                can_be_link = True
                re.search(r"(h?t?t?p?s?://)?(w{0,3}\.)?[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]
        except:
            # print("Not link!! " + words[index])
            can_be_link = False

        try:
            if '@' in words[index]:
                can_be_email = True
                re.search(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]
        except:
            # print("Not email!! " + words[index])
            can_be_email = False

        if can_be_email == False and can_be_link == False and len(str(words[index])) > 2 and is_normal_word(words[index]) == False:
            words[index] = "not_normal_word"
        elif can_be_email == False and can_be_link == False and len(str(words[index])) > 150:
            new_word = list()
            new_word.append("vvv_long_word")
            words = words + new_word
            # print("LONG WORD " + words[index])
        elif can_be_email == False and can_be_link == False and len(str(words[index])) > 100:
            new_word = list()
            new_word.append("vv_long_word")
            words = words + new_word
            # print("LONG WORD " + words[index])
        elif can_be_email == False and can_be_link == False and len(str(words[index])) > 70:
            new_word = list()
            new_word.append("v_long_word")
            words = words + new_word
            # print("LONG WORD " + words[index])
        elif can_be_email == False and can_be_link == False and len(str(words[index])) > 50:
            new_word = list()
            new_word.append("long_word")
            words = words + new_word
            # print("LONG WORD " + words[index])
        elif item.isalpha() == False:
            # print("NON-ALPHA " + words[index])
            if words[index].isalpha() == False:
                if len(words[index]) >= 1 + 4 + str(words[index]).startswith('$') and str(words[index])[
                    1].isnumeric():
                    words[index] = "$$$"
                elif len(words[index]) >= 4 + 4 and str(words[index]).startswith('usd$') and str(words[index])[
                    1].isnumeric():
                    words[index] = "usd$$$"
                    new_word = list()
                    new_word.append("$$$")
                    words = words + new_word
                elif len(words[index]) >= 1 + 2 and str(words[index])[1].isnumeric() and str(words[index]).startswith('$'):
                    words[index] = "$$"
                elif len(words[index]) >= 1 + 1 and str(words[index])[1].isnumeric() and str(words[index]).startswith('$'):
                    words[index] = "$"
                elif can_be_email == True:
                    words[index] = re.search(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]
                    # print(">CONTINE email " + words[index])

                    new_word = list('')
                    new_word.append(re.search(r"@[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0][1:])
                    words = words + new_word

                    new_word = list('')
                    new_word.append(re.search(r"[a-z0-9\.\-+_]+@", words[index])[0][0:-1])
                    words = words + new_word
                    #print("SENDER + " + new_word[0] + " _>" + words[index])

                elif ':' in words[index] and is_time(words[index]) != False:
                    words[index] = is_time(words[index])
                    new_word = list('')
                    new_word.append("found_time")
                    words = words + new_word
                    # print("Found time!!! " + words[index])
                elif len(str(words[index])) == 4 and is_year(words[index]) != False:
                    words[index] = is_year(words[index])
                    new_word = list('')
                    new_word.append("found_year")
                    words = words + new_word
                    # print("Found year!!! " + words[index])
                elif len(str(words[index])) > 2 and str(words[index])[0:2].isnumeric() and is_date(words[index]) != False:
                    words[index] = is_date(words[index])
                    new_word = list('')
                    new_word.append("found_year")
                    words = words + new_word
                    # print("Found year!!! " + words[index])
                elif can_be_link == True:
                    # print(">CONTINE link " + words[index])
                    words[index] = re.search(r"(h?t?t?p?s?://)?(w{0,3}\.)?[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]

                    # print(">CONTINE link1 " + words[index])
                    if urlparse(words[index]).netloc:
                        domain = urlparse(words[index])
                        if '.' in domain.netloc:
                            if domain.netloc.startswith('w') and 'w.' in str(domain.netloc):
                                words[index] = domain.scheme + "://" + domain.netloc.split('.')[0] + "." + \
                                               domain.netloc.split('.')[-2] + "." + \
                                               domain.netloc.split('.')[-1]
                            else:
                                words[index] = domain.scheme + "://" + domain.netloc.split('.')[-2] + "." + \
                                               domain.netloc.split('.')[-1]

                            # print(">CONTINE link2 " + domain.netloc.split('.')[-2] + "." + domain.netloc.split('.')[-1])
                            # adaug domeniul
                            new_word = list()
                            new_word.append(domain.netloc.split('.')[-2] + "." + domain.netloc.split('.')[-1])
                            words = words + new_word

                            # adaug domeniul tarii
                            new_word = list()
                            new_word.append("." + domain.netloc.split('.')[-1])
                            words = words + new_word

                            # adaug http
                            new_word = list()
                            new_word.append(domain.scheme)
                            words = words + new_word
                        else:
                            words[index] = domain.netloc

                        # print(">CONTINE link3 " + words[index])

                    else:
                        new_word = list()
                        new_word.append("no_http")
                        words = words + new_word
                        # print("NO HTTP!" + words[index])
                        if words[index].startswith('w') and 'w.' in words[index]:
                            words[index] = words[index].split('.')[0] + "." + \
                                           words[index].split('.')[-2] + "." + \
                                           words[index].split('.')[-1]
                        else:
                            words[index] = words[index].split('.')[-2] + "." + words[index].split('.')[-1]

                        new_word = list()
                        new_word.append(words[index].split('.')[-2] + "." + words[index].split('.')[-1])
                        words = words + new_word
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
                        words[index] = ' '
    return words


def make_dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    print("Making dictionary", end ="", flush=True)
    for mail in emails:
        with open(mail, encoding="Latin-1") as m:
            print(".", end ="", flush=True)
            for i, line in enumerate(m):
                #if i == 2:  # Body of email is only 3rd line of text file
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
    new_dictionary=dict()
    for i,item in enumerate(dictionary):
        new_dictionary[item[0]]=(i,item[1])



    print("\n", flush=True)
    # print(new_dictionary)
    return new_dictionary


def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), DICTIONARY_SIZE))
    docID = 0
    fileID = 0
    print("\nExtracting features", end ="", flush=True)
    for file in files:
        print(".", end ="", flush=True)
        fileID = fileID + 1
        with open(file, encoding="Latin-1") as fi:
            for i, line in enumerate(fi):
                words = line.split()
                process_words(words)
                for word in words:
                        if word in dictionary:
                            wordID = dictionary[word][0]
                            features_matrix[docID, wordID] = features_matrix[docID, wordID] + words.count(word)
            docID = docID + 1
    print("\n", flush=True)
    return features_matrix


def extract_features_train(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), DICTIONARY_SIZE))
    email_type = np.zeros((len(files)))
    docID = 0
    fileID = 0
    print("\nExtracting features", end ="", flush=True)
    for file in files:
        print(".", end ="", flush=True)
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
                words = line.split()
                process_words(words)
                for word in words:
                        if word in dictionary:
                            wordID = dictionary[word][0]
                            features_matrix[docID, wordID] = features_matrix[docID, wordID] + words.count(word)
            docID = docID + 1
    print("\n", flush=True)
    return features_matrix, email_type


def build_model(x_train, y_train):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    return model


def classify_emails_train(mail_dir, model, x_test, output_file):
    f = open(output_file, "w")
    predictions = model.predict([x_test])

    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]

    nr_of_misses=0
    for i, file in enumerate(files):
        if predictions[i] > 0.5:
            f.write(str(i) + ". " + file + "|inf" + " " + str(predictions[i]))
            if file.endswith(".cln"):
                nr_of_misses = nr_of_misses +1
                f.write("MISS!")
        elif predictions[i] <= 0.5:
            f.write(str(i) + ". " + file + "|cln" + " " + str(predictions[i]))
            if file.endswith(".inf"):
                nr_of_misses = nr_of_misses +1
                f.write("MISS!")
        else:
            f.write("Classification error!")
        f.write("\n")
    f.close()
    print("Misses: " + str(nr_of_misses))

def classify_emails(mail_dir, model, x_test, output_file):
    f = open(output_file, "w")
    predictions = model.predict([x_test])
    for i, file in enumerate(os.listdir(mail_dir)):
        if predictions[i] > 0.5:
            f.write(file + "|inf")
        elif predictions[i] <= 0.5:
            f.write(file + "|cln")
        f.write("\n")
    f.close()

if len(sys.argv) == 1:
    print("Run it with arguments!")
elif sys.argv[1] == "-info":
    f = open(sys.argv[2], "w")
    f.write("Anti_Spam_Filter_SSOSM\n")
    f.write("Haloca_Dorin\n")
    f.write("PaleVader\n")
    f.write("Version_1.1\n")
    f.close()
elif sys.argv[1] == "-scan":
    dictionary=dict()
    with open('dictionary.json', 'r') as fp:
        dictionary=json.load(fp)
    model=tf.keras.models.load_model("spam_filter.model", compile=False)
    x_test = extract_features(sys.argv[2], dictionary)
    classify_emails(sys.argv[2], model, x_test, sys.argv[3])

elif sys.argv[1] == "-train":
    dictionary = make_dictionary(sys.argv[2])
    x_train, y_train = extract_features_train(sys.argv[2], dictionary)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    model = build_model(x_train, y_train)
    model.save("spam_filter.model")

    with open('dictionary.json', 'w') as fp:
        json.dump(dictionary, fp)

elif sys.argv[1] == "-train+scan":
    verdict = "cln"

    dictionary = make_dictionary(sys.argv[2])
    x_train, y_train = extract_features_train(sys.argv[2], dictionary)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    model = build_model(x_train, y_train)

    x_test, y_test = extract_features_train(sys.argv[3], dictionary)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    val_loss, val_acc = model.evaluate(x_test, y_test)
    classify_emails_train(sys.argv[3], model, x_test, sys.argv[4])
    # print("loss: " + str(val_loss) + " - acc: " + str(val_acc))

    f = open("dictionary.txt", "w")
    f.write(str(dictionary))
    f.close()

    # f_read.close()
    # if verdict == "inf":
    # print(os.path.join(sys.argv[2], file + "|" + verdict))
    # f_out.write(file.replace(" ", "_") + "|" + verdict)
    # f_out.write("\n");
