import csv
import csv, codecs
import collections
from collections import Counter
import gensim
import logging
#import nltk
import codecs
import numpy as np
import math
from scipy import linalg, dot
import gensim, logging
import lda
import itertools
import logging, gensim, bz2
from gensim import corpora, models, similarities
from gensim.models import hdpmodel, ldamodel
from itertools import izip

def clean_comments(comments):

    items = []
    for item in comments:

        item = item.replace('"', '')
        item = ' '.join(item.split())
        item = item.lower()
        item = item.replace(".", "")
        item = item.replace(",", "")
        item = item.replace("'", "")
        item = item.replace("(", " ")
        item = item.replace(")", " ")
        item = item.replace("/", " ")
        item = item.replace("-", " ")
        item = item.replace("ve", " ")
        item = item.replace("ama", " ")
        item = item.replace("herhangi", " ")
        item = item.replace("baska", " ")
        item = item.replace("boyle", " ")
        item = item.replace("her", " ")
        item = item.replace("cogu", " ")
        item = item.strip()

        items.append(item)
    return items

def word2vec (sentence):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    item_list = []
    for item in sentence:
        item = item.split()
        item_list.append(item)


    model = gensim.models.Word2Vec(item_list, min_count=10)

    # return model.most_similar("memnuniyetsizliklerini")
    # return model.most_similar("eksik")
    # return model.most_similar("yemek")
    # return model.most_similar("ikram")
    # return model.most_similar("tavuk")
    # return model.most_similar("balik")
    # return model.most_similar("salata")
    # return model.most_similar("yuklenmemisti")
    # return model.most_similar("seferimizde")
    # return model.most_similar("menu")
    # return model.most_similar("yolcumuz")
    # return model.most_similar("cocuk")
    # return model.most_similar("gazete")
    # return model.most_similar("koltukta")
    # return model.most_similar("talep")
    return model.most_similar("bozuk")


#
# def lda(sentence):
#
#     logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
#     documents = list(itertools.chain(*sentence))
#
#     item_list = []
#     for item in sentence:
#         item = item.split()
#         item_list.append(item)
#
#     texts = item_list
#
#     # remove words that appear only once
#     all_tokens = sum(texts, [])
#     tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
#     texts = [[word for word in text if word not in tokens_once]
#              for text in texts]
#
#
#     vocab_list = list(itertools.chain(*item_list))
#
#     lda = gensim.models.ldamodel.LdaModel(corpus=vocab_list, id2word= vocab_list,  num_topics=100, update_every=1, chunksize=10000, passes=1)
#     return lda.print_topics(20)





if __name__=='__main__':

    filename1 = '/home/stm/PycharmProjects/thy_kabin/kabin_raporu_2015.csv'
    filename2 = '/home/stm/PycharmProjects/thy_kabin/kabin_raporu_2015_devam.csv'


    with open(filename1) as file:
        file.next()
        train_reader = csv.reader(file, delimiter=';')

        dates = []
        chulk_number = []      #sefer numarasi
        departure_port = []    #kalkis limani
        arrival_port = []      #varis limani
        queue_names = []       #kuyruk adlari
        passenger_number = []  #toplam yolcu sayisi
        cabin_class = []       #ucak kabin siniflari
        unit = []
        categories = []        #kategoriler
        subjects =  []         #konular
        detail = []            #detay
        explanation = []       #aciklamalar
        evaluation = []        #degerlendirme
        situation = []         #durum


        for rows in train_reader:
            tarih = rows[0]
            sefer_no = rows[1]
            kalkis_liman = rows[2]
            varis_liman = rows[3]
            kuyruk_ad = rows[4]
            toplam_yolcu = rows[5]
            ucak_kabin_sinif = rows[6]
            birim = rows[7]
            kategori = rows[8]
            konu = rows[9]
            detay = rows[10]
            aciklama = rows[11]
            degerlendirme = rows[12]
            durum = rows[13]

            dates.append(tarih)
            chulk_number.append(sefer_no)
            departure_port.append(kalkis_liman)
            arrival_port.append(varis_liman)
            queue_names.append(kuyruk_ad)
            passenger_number.append(toplam_yolcu)
            cabin_class.append(ucak_kabin_sinif)
            categories.append(kategori)
            subjects.append(konu)
            detail.append(detay)
            explanation.append(aciklama)
            evaluation.append(degerlendirme)
            situation.append(durum)


        cleaned_explanation = clean_comments(explanation)
        cleaned_subjects = clean_comments(subjects)
        cleaned_evaluation = clean_comments(evaluation)
        cleaned_detail = clean_comments(detail)


        print word2vec(cleaned_explanation)


