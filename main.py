import tensorflow as tf
from urllib.parse import urlparse
import numpy as np
import os
import sys
from collections import Counter
import re
import json
from shutil import copyfile
import base64

DICTIONARY_SIZE = 7000
DICTIONARY_RESERVED = 100  # Words selected by me
SPECIAL_WORDS = ["base64", "hi!", "hi!!", "hi!!!", "subject:hi!", "subject:hi!!",
                 "subject:hi!!!"]  # Avoids processing (symbols removal etc.)


def is_normal_word(word):
    # try:
    #    re.search(r"[a-z0-9></]", word)[0]
    # except:
    #    return False

    nr = 0
    for item in word:
        if item in "abcdefghijklmnopqrstuvwxyz":
            nr = nr + 1
        #elif item in "0123456789":
        #    nr = nr + 1
    if nr == 0 or len(word) / nr > 4:
        return False

    return True


def process_words(words, debug=False):
    if len(words) < 1:
        return words

    allow_normal_words = False
    if debug == True:
        print("1:" + str(words))
    if words[0].lower().startswith("subject"):
        words[0] = words[0][8:]
        if len(words[0]) == 0 and len(words) == 1:
            words[0] = "no_subject"
            return words
        allow_normal_words = True
        for index, item in enumerate(words):
            words_list = re.split('(\?|_|=|[|])', item)
            words[index] = ' '
            for words_list_item in words_list:
                if len(words_list_item) > 1:
                    new_word = list()
                    new_word.append("subject:" + words_list_item)
                    words = words + new_word

                    new_word = list()
                    new_word.append(words_list_item)
                    words = words + new_word
    elif len(words) == 1 and len(words[0]) == 76:
        try:
            decoded = base64.b64decode(words[0][:4])
            if str(base64.b64encode(decoded))[2:-1] == words[0][:4]:  # Shorter test to speed up the process
                decoded = base64.b64decode(words[0])
                if str(base64.b64encode(decoded))[2:-1] == words[0]:
                    return "---base64"
        except:
            pass

    for index, item in enumerate(words):

        if len(words[index]) <= 2:
            words[index] = ' '
            continue

        words[index] = words[index].lower()

        if words[index].isalpha() == True or words[index] in SPECIAL_WORDS:
            continue

        can_be_link = False
        can_be_email = False
        link_search = str()
        email_search = list()
        try:
            if 'w.' in words[index] or ":/" in words[index]:
                can_be_link = True
                link_search = re.search(r"(h?t?t?p?s?://)?(w{0,3}\.)?[a-z0-9\.\-+_]+\.[a-z]+", words[index])[0]
        except:
            # print("Not link!! " + words[index])
            can_be_link = False

        try:
            if '@' in words[index] and '.' in words[index]:
                can_be_email = True
                email_search = re.findall(
                    r"(([a-z0-9\.\-+_]+@[x\-+_]+\.[x]{2,3})|[a-z0-9\.\-+_]+@[a-z0-9\-+_]+\.[a-z]{2,3})", words[index])
        except:
            # print("Not email!! " + words[index])
            can_be_email = False

        if debug == True:
            print("2.email:" + str(can_be_email))

        # if can_be_email == False and can_be_link == False:
        #    words[index] = words[index].lstrip('\\xef\\xbb\\xbf')
        #    words[index] = words[index].rstrip('\\r\\n')

        if can_be_email == False and can_be_link == False:
            words_index = words[index]
            words_index_first = words[index][0]
            words_index_last = words[index][-1]
            if words_index_first == '<' or words_index_last == '>' or words_index_first == '\\' or \
                    words_index_first == '/' or words_index_first == '-':
                words[index] = ' '
                continue
            if words_index_last == '\"' and words_index_first != '\"':
                words[index] = ' '
                continue
            if '\'' in words_index:
                words[index] = words_index.rstrip('\'')
                words[index] = words_index.lstrip('\'')
            if '\"' in words_index:
                words[index] = words_index.rstrip('\"')
                words[index] = words_index.lstrip('\"')
            if words_index_last == '.':
                words[index] = words_index.rstrip('.')
            elif words_index_last == ',':
                words[index] = words_index.rstrip(',')
            elif words_index_last == '!':
                words[index] = words_index.rstrip('!')
            elif words_index_last == '?':
                words[index] = words_index.rstrip('?')
            elif words_index_last == ':':
                words[index] = words_index.rstrip(':')
            elif words_index_last == ';':
                words[index] = words_index.rstrip(';')
            if words_index_first == '[':
                words[index] = words_index.lstrip('[')
            if words_index_last == ']':
                words[index] = words_index.rstrip(']')
            if words_index_first == '(':
                words[index] = words_index.lstrip('(')
            if words_index_last == ')':
                words[index] = words_index.rstrip(')')
            # words[index] = words_index.replace('-', '')

            if len(words[index]) <= 2:
                words[index] = ' '
                continue

            if allow_normal_words == True and is_normal_word(words[index]) == True:
                continue
            if words[index].isalpha() == False:
                words[index] = ' '
        else:
            if can_be_email == True:
                # Sa gaseasca adresa cu xxxx... si sa aleaga doar primii xx dupa ., nu xxa, sau ceva de genul
                matches = email_search

                if debug == True:
                    print("3.email:" + str(matches))

                # print(">CONTINE email " + words[index])
                words[index] = ' '
                for match in matches:
                    match = match[0]
                    # if "xxx" in match:
                    # print("XXX:" + match)
                    # print(words[index])

                    new_word = list('')
                    new_word.append(re.search(r"(@[x\-+_]+\.[x]{2,3}|@[a-z0-9\-+_]+\.[a-z]{2,3})", match)[0][1:])
                    words = words + new_word

                    new_word = list('')
                    new_word.append(re.search(r"[a-z0-9\.\-+_]+@", match)[0][0:-1])
                    words = words + new_word
                    # print("SENDER + " + new_word[0] + " _>" + words[index])
                    continue
            elif can_be_link == True:

                # print(">CONTINE link " + words[index])
                words[index] = link_search

                # print(">CONTINE link1 " + words[index])
                url_parse = urlparse(words[index])
                if url_parse.netloc:
                    domain = url_parse
                    if '.' in domain.netloc:
                        if domain.netloc.startswith('w') and 'w.' in domain.netloc:
                            words_split = domain.netloc.split('.')
                            words[index] = domain.scheme + "://" + words_split[0] + "." + \
                                           words_split[-2] + "." + \
                                           words_split[-1]
                        else:
                            words_split = domain.netloc.split('.')
                            words[index] = domain.scheme + "://" + words_split[-2] + "." + \
                                           words_split[-1]

                        # print(">CONTINE link2 " + domain.netloc.split('.')[-2] + "." + domain.netloc.split('.')[-1])
                        # adaug domeniul
                        split_netloc = domain.netloc.split('.')
                        new_word = list()
                        words_split = split_netloc
                        new_word.append(words_split[-2] + "." + words_split[-1])
                        words = words + new_word

                        # adaug domeniul tarii
                        new_word = list()
                        new_word.append("." + split_netloc[-1])
                        words = words + new_word

                        # adaug http
                        new_word = list()
                        new_word.append(domain.scheme)
                        words = words + new_word
                    else:
                        words[index] = domain.netloc

                    # print(">CONTINE link3 " + words[index])

                else:
                    # new_word = list()
                    # new_word.append("no_http")
                    # words = words + new_word
                    # print("NO HTTP!" + words[index])
                    if words[index].startswith('w') and 'w.' in words[index]:
                        words_split = words[index].split('.')
                        words[index] = words_split[0] + "." + \
                                       words_split[-2] + "." + \
                                       words_split[-1]
                    else:
                        words_split = words[index].split('.')
                        words[index] = words_split[-2] + "." + words_split[-1]

                    new_word = list()
                    words_split = words[index].split('.')
                    new_word.append(words_split[-2] + "." + words_split[-1])
                    words = words + new_word
                continue

            else:
                words[index] = ' '
    return words


def convert_to_base64(string):
    # print("BASE64:"+words[0])
    try:
        decoded = base64.b64decode(string)
        decoded = str(decoded)[2:-1]

        return decoded

    except:
        # print("ERROR")
        # print('')
        return "ERROR"


def make_dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    print("Making dictionary", end="", flush=True)

    for mail in emails:
        with open(mail, encoding="Latin-1") as m:
            print(".", end="", flush=True)
            base64 = False
            base64_string = str()
            for i, line in enumerate(m):
                words = line.split()
                # if "55fa89ceceb5f74a1e3e602b0415c57c" in mail:
                #    print(words)
                words = process_words(words)  # parse URL, remove non-alpha words
                if words == "---base64":
                    # print("ORIGINAL:"+str(line.split()))
                    if base64 == False:
                        base64 = True
                        base64_string = convert_to_base64(line.split()[0])
                        # print("DECODED:"+base64_string)
                        words = ' '
                    else:
                        base64_string = base64_string + convert_to_base64(line.split()[0])
                        # print("DECODED:" + base64_string)
                else:
                    if base64 == True:
                        base64 = False
                        if "55fa89ceceb5f74a1e3e602b0415c57c" in mail:
                            print(base64_string)
                        words = words + process_words(base64_string.split())
                if "55fa89ceceb5f74a1e3e602b0415c57c" in mail:
                    print(words)

                # if "55fa89ceceb5f74a1e3e602b0415c57c.inf" in mail:
                #    words = process_words(words, True)  # parse URL, remove non-alpha words

                # print(words)
                #    exit(0)
                # else:

                # if i == 0:
                #    words = test_file_size(words, mail)
                # if '2ca488083d53aa450085685ca4a48674' in mail:
                #    print(words)
                #    exit(0)

                all_words += words
                # if  find_spam_words(words) != None:
                #    print(find_spam_words(words))

    dictionary = Counter(all_words)
    list_to_remove = dictionary.keys()
    for item in list(list_to_remove):
        if len(item) <= 1:  # caut spatiu si cuvinte nule
            del dictionary[item]

    dictionary = dictionary.most_common(DICTIONARY_SIZE - DICTIONARY_RESERVED)

    new_dictionary = dict()
    for i, item in enumerate(dictionary):
        new_dictionary[item[0]] = (i, item[1])
    with open("reserved_words.txt") as m:
        for i, line in enumerate(m):
            line = line.rstrip('\n')
            if line not in new_dictionary:
                new_dictionary[line] = (DICTIONARY_SIZE - DICTIONARY_RESERVED + 1 + i, 1)

    print("\n", flush=True)
    # print(new_dictionary)
    return new_dictionary


def test_file_size(words, mail):
    file_size = os.path.getsize(mail)
    if file_size / 1000 > 300:
        new_word = list()
        new_word.append("vvvv_big_file")
        words = words + new_word
    if file_size / 1000 > 200:
        new_word = list()
        new_word.append("vvv_big_file")
        words = words + new_word
    if file_size / 1000 > 150:
        new_word = list()
        new_word.append("vv_big_file")
        words = words + new_word
    if file_size / 1000 > 100:
        new_word = list()
        new_word.append("v_big_file")
        words = words + new_word
    elif file_size / 1000 > 50:
        new_word = list()
        new_word.append("big_file")
        words = words + new_word
    elif file_size < 1500:
        new_word = list()
        new_word.append("small_file")
        words = words + new_word
    elif file_size < 1000:
        new_word = list()
        new_word.append("v_small_file")
        words = words + new_word

    return words


def extract_features(mail_dir, dictionary):
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), DICTIONARY_SIZE))
    docID = 0
    fileID = 0
    print("\nExtracting features...", end="", flush=True)
    for file in files:
        print(".", end="", flush=True)
        fileID = fileID + 1
        with open(file, encoding="Latin-1") as fi:
            base64 = False
            base64_string = str()
            for i, line in enumerate(fi):
                words = line.split()
                words = process_words(words)  # parse URL, remove non-alpha words
                if words == "---base64":
                    if base64 == False:
                        base64 = True
                        base64_string = convert_to_base64(line.split()[0])
                        words = ' '
                    else:
                        base64_string = base64_string + convert_to_base64(line.split()[0])
                else:
                    if base64 == True:
                        base64 = False
                        words = words + process_words(base64_string.split())

                # if i == 0:
                #    words = test_file_size(words, file)

                for word in words:
                    if word != ' ' and word in dictionary:
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
    print("\nExtracting features", end="", flush=True)
    for file in files:
        print(".", end="", flush=True)
        if file.endswith(".cln"):
            email_type[fileID] = 0
        elif file.endswith(".inf"):
            email_type[fileID] = 1
        else:
            print("Invalid email type for training!")
            exit(-1)
        fileID = fileID + 1
        with open(file, encoding="Latin-1") as fi:
            print(".", end="", flush=True)
            base64 = False
            base64_string = str()
            for i, line in enumerate(fi):
                words = line.split()
                words = process_words(words)  # parse URL, remove non-alpha words
                if words == "---base64":
                    if base64 == False:
                        base64 = True
                        base64_string = convert_to_base64(line.split()[0])
                        words = ' '
                    else:
                        base64_string = base64_string + convert_to_base64(line.split()[0])
                else:
                    if base64 == True:
                        base64 = False
                        words = words + process_words(base64_string.split())
                # if i == 0:
                #    words = test_file_size(words, file)

                for word in words:
                    if word != ' ' and word in dictionary:
                        wordID = dictionary[word][0]
                        features_matrix[docID, wordID] = features_matrix[docID, wordID] + words.count(word)

            docID = docID + 1
    print("\n", flush=True)
    return features_matrix, email_type


def build_model(x_train, y_train):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(DICTIONARY_SIZE,)))
    model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(300, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # print("\n TIPPPPPP" + str(type(x_train)) + "\n ")
    model.fit(x_train, y_train, epochs=3)
    return model


def classify_emails_train(mail_dir, model, x_test, output_file):
    # print("\n TIPPPPPP" + str(type(x_test)) + "\n ")
    f = open(output_file, "w")
    predictions = model.predict(x_test)
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    nr_of_misses = 0

    for i, file in enumerate(files):
        if predictions[i] > 0.5:
            f.write(str(i) + ". " + file + "|inf" + " " + str(predictions[i]))
            if file.endswith(".cln"):
                nr_of_misses = nr_of_misses + 1
                f.write("MISS!")
                copyfile(file, os.path.join("Problem", file.split('\\')[-1]))
        elif predictions[i] <= 0.5:
            f.write(str(i) + ". " + file + "|cln" + " " + str(predictions[i]))
            if file.endswith(".inf"):
                nr_of_misses = nr_of_misses + 1
                f.write("MISS!")
                print(file)
                copyfile(file, os.path.join("Problem", file.split('\\')[-1]))
        else:
            f.write("Classification error!")
        f.write("\n")
    f.close()
    print("Misses: " + str(nr_of_misses))


def classify_emails(mail_dir, model, x_test, output_file):
    # print("\n TIPPPPPP" + str(type(x_test)) + "\n ")
    f = open(output_file, "w")
    predictions = model.predict(x_test)

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
    f.write("Version_2.03\n")
    f.close()
elif sys.argv[1] == "-scan":
    dictionary = dict()
    with open('dictionary.json', 'r') as fp:
        dictionary = json.load(fp)
    model = tf.keras.models.load_model("spam_filter.h5", compile=True)
    x_test = extract_features(sys.argv[2], dictionary)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    # print("\n TIPPPPPP - normalised_test" + str(type(x_test)) + "\n ")
    classify_emails(sys.argv[2], model, x_test, sys.argv[3])

elif sys.argv[1] == "-scan_prob":
    dictionary = dict()
    with open('dictionary.json', 'r') as fp:
        dictionary = json.load(fp)
    model = tf.keras.models.load_model("spam_filter.h5", compile=True)
    x_test = extract_features(sys.argv[2], dictionary)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    # print("\n TIPPPPPP - normalised_test" + str(type(x_test)) + "\n ")
    classify_emails_train(sys.argv[2], model, x_test, sys.argv[3])

elif sys.argv[1] == "-train":
    dictionary = make_dictionary(sys.argv[2])
    x_train, y_train = extract_features_train(sys.argv[2], dictionary)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    # print("\n TIPPPPPP - normalised" + str(type(x_train)) + "\n ")
    model = build_model(x_train, y_train)
    model.save('spam_filter.h5', save_format='h5')

    with open('dictionary.json', 'w') as fp:
        json.dump(dictionary, fp)

elif sys.argv[1] == "-train+scan":
    verdict = "cln"

    dictionary = make_dictionary(sys.argv[2])
    x_train, y_train = extract_features_train(sys.argv[2], dictionary)
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    # print("\n TIPPPPPP - normalised_train" + str(type(x_train)) + "\n ")
    model = build_model(x_train, y_train)

    x_test, y_test = extract_features_train(sys.argv[3], dictionary)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    # print("\n TIPPPPPP - normalised_test" + str(type(x_test)) + "\n ")
    val_loss, val_acc = model.evaluate(x_test, y_test)
    classify_emails_train(sys.argv[3], model, x_test, sys.argv[4])
    # print("loss: " + str(val_loss) + " - acc: " + str(val_acc))

    f = open("dictionary.txt", "w")
    f.write(str(dictionary))
    f.close()

else:
    print("Invalid argument!")
    # f_read.close()
    # if verdict == "inf":
    # print(os.path.join(sys.argv[2], file + "|" + verdict))
    # f_out.write(file.replace(" ", "_") + "|" + verdict)
    # f_out.write("\n");
