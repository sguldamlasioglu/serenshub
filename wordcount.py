
# coding: utf-8

# In[1]:

import sys
from operator import add

data = sc.textFile("hdfs://bee13:8020/user/thy/cleaned_explanation_all")
data.count()

   


# In[2]:

data2 = data.map(lambda x: x.split())
print data2.take(2)
print data2.count()


# In[3]:

data3 = data2.flatMap(lambda x: x)
print data3.count()


# In[4]:

data4 = data3.map(lambda x:(x,1))
print data4.take(4)
data5 = data4.reduceByKey(add)
print data5.take(2)
print data5.count()


# # data6 = data5.map(lambda x: (x[1],x[0])).sortByKey()
# print data6.top(200)

# In[11]:

top_items = ['yuklemesi', 'yolcumuz', 'bc', 'yemek', 'adet', 'yolcu', 'ikram', 'yuklenen', 'servis', 'sicak', 'yuklenmemistir', 'menu', 'yuklenmistir', 'yetersiz', 'eksik', 'ist', 'film', 'yemegi', 'tavuk', 'yapilmamistir', 'suyu', 'yukleme', 'ariza', 'sistem', 'talep', 'bardak', 'olmadigi', 'gazete', 'lokum', 'koltugunda', 'kahvalti', 'cocuk', 'makarna', 'tepsi','siparis', 'cay', 'secenek' ]


# In[ ]:



