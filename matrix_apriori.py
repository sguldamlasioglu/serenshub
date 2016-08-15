
# coding: utf-8

# In[6]:

data = sc.textFile("hdfs://bee13:8020/user/thy/cleaned_explanation_all")
data.take(3)


# In[2]:

top_items = [u'ikram', u'sicak',u'eksik', u'yetersiz' , u'film' , u'tavuk', u'bardak', u'gazete', u'lokum', u'kahvalti', u'cocuk', u'makarna', u'tepsi', u'cay', u'soguk', u'kahve', u'portakal', u'kofte', u'sarap',  u'oyuncak']


# In[3]:

### all list (list of lists)
splitted = data.map(lambda line: line.strip().split(','))
print splitted.take(5)
print splitted.count()


# In[4]:

## all words in lists
flat = splitted.flatMap(lambda x:x)
splitted_words = flat.map(lambda line: line.strip().lower().split(' '))
print splitted_words.take(10)
print splitted_words.count()


# In[7]:

def func1 (row,cols):
    colValList = []
    for i in cols:
        if any(i in s for s in row):
            colValList.append(1)
        else:
            colValList.append(0)
    
    return colValList

def func2(x):
    strRet = ""
    for i in x:
        strRet = strRet + str(i) + ','
    strRet = strRet[:-1]
    return strRet

rdd2 = splitted_words.map(lambda x: func1(x,top_items))
print rdd2.count()
print rdd2.take(3)

rdd3 = rdd2.map(lambda x: func2(x))
print rdd3.count()
print rdd3.take(3)

rdd3.saveAsTextFile("hdfs://bee13:8020/user/thy/akyoutputxx66")


# In[15]:



