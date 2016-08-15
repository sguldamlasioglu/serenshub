
# coding: utf-8

# In[1]:

import sys
from operator import add
from pyspark.mllib.feature import Word2Vec

data = sc.textFile("hdfs://bee13:8020/user/xxx.csv")
data.count()



# In[14]:

from pyspark.ml.feature import Word2Vec

documentDF = sqlContext.createDataFrame(data.map(lambda row: [row.split(" ")]), ["text"])

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=10, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)
result = model.transform(documentDF)

for feature in result.select("result").take(3):
    print feature

