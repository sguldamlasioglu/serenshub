
# coding: utf-8

# In[2]:

from pyspark.mllib.fpm import FPGrowth


# In[3]:

data = sc.textFile("hdfs://bee13:8020/user/thy/kabin_raporu_2015.csv")
data.first()


# In[4]:

import sys
import traceback

def formatResult(result):
    
    try:
        return result[11]
    except:
        return "NULL"

    


# In[5]:

def distinct_set(list):

    mySet = set(list)
    
    return mySet


# In[6]:

def distinct_list(set):

    mylist = list(set)
    
    return mylist


# In[7]:

splitted = data.map(lambda line: line.strip().split(';'))
explanation = splitted.map(lambda x :  formatResult(x))
print explanation.take(2)


# In[8]:

splitted_explanation = explanation.map(lambda line: line.strip().lower().split(' '))
print splitted_explanation.take(5)


# In[12]:

distincted_set = splitted_explanation.map(lambda line: distinct_set(line))
distincted_set.take(5)


# In[13]:

distinctedlist = distincted_set.map(lambda line: distinct_list(line))
distinctedlist.take(5)


# In[14]:

from pyspark.mllib.fpm import FPGrowth
model = FPGrowth.train(distinctedlist, minSupport=0.001, numPartitions=1000)

result = model.freqItemsets().collect()

for fi in result:
    print(fi)


# In[ ]:



