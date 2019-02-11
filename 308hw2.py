# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 01:07:16 2019

@author: Amy
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder



storeinfo = pd.read_csv('C:\\Users\\Amy\\Downloads\\\Dillards POS\\Dillards POS\\strinfo.csv',header=None)
storeinfo.head(5)
storeinfo.columns = ['Storenum', 'City', 'State','Zip','unknown']
storeinfo.groupby(['State'])['Storenum'].count()
#transdf=pd.read_csv('C:\\Users\\Amy\\Downloads\\\Dillards POS\\Dillards POS\\trnsact.csv',sep=',',chunksize=100000)
num_lines = sum(1 for line in open('C:\\Users\\Amy\\Downloads\\\Dillards POS\\Dillards POS\\trnsact.csv'))
print (num_lines)
#study supermarkets in nebraska 4 stores
storeil=storeinfo[storeinfo.State=='IL']
print (storeil) #storenum is 603,1003,4903
storevaluelist=['603','1003','4903']


df_empty=pd.DataFrame()
chunkdf=pd.read_csv('C:\\Users\\Amy\\Downloads\\\Dillards POS\\Dillards POS\\trnsact.csv',header=None,chunksize =2000000)
for chunk in chunkdf:
    chunk = chunk[chunk[1].isin(storevaluelist)]
    chunk = chunk[chunk[6]=='P']
    chunk=chunk[[0,1,2,3,5,6,7,11,12]]
    df_empty=pd.concat([df_empty,chunk])
    
    
df_empty.to_csv('cleaningupdata.csv')

df = df_empty.copy()
df.columns=['SKU','Store','register','transaction','date','status','quantity','interid','mic']

del df['status']

valuecount = pd.DataFrame(df['SKU'].value_counts())
valuecount.head(10)
minisupportdf=pd.DataFrame(valuecount[valuecount['SKU']>=20]).index.tolist()


newdf = df[df.SKU.isin(minisupportdf)]


#plt.hist(df['SKU'])
#plt.show()
# =============================================================================
# 
# list_of_baskets=[]
# grouped_trial = df.groupby(['Store','register','transaction'])
# for name,group in grouped_trial:
# 
#     (u_s, u_r,u_t)=name
#     basket_SKUs = group['SKU'].unique().tolist()
#     if len(basket_SKUs) > 1:
#          d = {'store':u_s, 'register':u_r, 'trannum':u_t, 'basket':basket_SKUs,'date':u_date}
#          list_of_baskets.append(d)
#     else:
#          pass
# =============================================================================
     
list_of_baskets=[]
grouped_trial = newdf.groupby(['Store','register','transaction','date'])
for name, group in grouped_trial:
    (u_s, u_r, u_t,u_date) = name
    basket_SKUs = group['SKU'].unique().tolist()
    if len(basket_SKUs) > 1:
        d = {'store':u_s, 'register':u_r, 'trannum':u_t, 'basket':basket_SKUs,'date':u_date}
        list_of_baskets.append(d)
    else:
        pass
    
basket_df = pd.DataFrame(list_of_baskets)
testbasket= basket_df.copy()
testbasket=testbasket.set_index(pd.DatetimeIndex(testbasket['date']))
g=testbasket.groupby(pd.Grouper(freq="M"))
g.count()


# set a minimum support rule

# =============================================================================
# for trial purpose 
# mask = (basket_df['date']>'2004-11-01') & (basket_df['date']<= '2004-11-30')
# basketnewdf=basket_df.loc[mask]
# =============================================================================
#basketnewdf= basket_df.loc[basket_df['date'].isin(['2004-12-01','2005-08-27'])] it doens't work
all_baskets = basket_df['basket'].tolist()

te=TransactionEncoder()
te_array=te.fit(all_baskets).transform(all_baskets)
sparsedf = pd.DataFrame(te_array,columns=te.columns_)

sparsetestdf=sparsedf.copy()


frequent_itemsets=apriori(sparsetestdf,min_support=0.0005,use_colnames=True)
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=1)


rulescopy = rules.copy()
antecedentlist=[]
consequentlist=[]
for index,row in rulescopy.iterrows():
    ante=list(row['antecedents'])
    conse=list(row['consequents'])
    antecedentlist.append(ante)
    consequentlist.append(conse)
rulescopy['trueantecedents']=antecedentlist
rulescopy['trueconsequents']=consequentlist
del rulescopy['antecedents']
del rulescopy['consequents']

rulescopy=rulescopy[['trueantecedents','trueconsequents','support','confidence','lift','leverage']]

finalrules = rulescopy.copy()
finalrules = finalrules.sort_values(by=['confidence'],ascending=False)
finalrules100 = finalrules.iloc[:100]
finals20 = finalrules.iloc[:20]
finalrules100.to_csv('complete 100 rule.csv')
finals20.to_csv('top20rules.csv')
# =============================================================================
tags = finalrules['trueantecedents'].apply(pd.Series)
tags = tags.rename(columns = lambda x : 'tag_' + str(x))
tags.columns=['sku','sku2']
tags2 = finalrules['trueconsequents'].apply(pd.Series)
tags2 = tags2.rename(columns = lambda x : 'tag2_' + str(x))
tags2.columns=['sku','sku2']
rulesunlist=pd.concat([tags,tags2],axis=1)

skuinfo=pd.read_csv('C:\\Users\\Amy\\Downloads\\\Dillards POS\\Dillards POS\\skuinfo.csv',header=None,usecols=[0,1])
skuinfo.columns=['sku','dept']

deptinfo=pd.read_csv('C:\\Users\\Amy\\Downloads\\\Dillards POS\\Dillards POS\\deptinfo.csv',header=None)
deptinfo.columns=['dept','description','unknwon']
newskuinfo = pd.merge(skuinfo,deptinfo,on='dept',how='left')
tags1merge=pd.merge(tags,newskuinfo,on='sku',how='left')
tags2merge = pd.merge(tags2,newskuinfo,on='sku',how='left')
tags12merge =pd.merge(tags,newskuinfo,left_on='sku2',right_on='sku',how='left')
tags22merge =pd.merge(tags2,newskuinfo,left_on='sku2',right_on='sku',how='left')
finalresult=pd.concat([tags1merge,tags12merge,tags2merge,tags22merge],axis=1)
del finalresult['unknwon']
del finalresult['sku_y']
finalresultcopy=finalresult.copy()
cols=[1,2,5,6,9,10,13,14]
finalresultcopy.drop(finalresultcopy.columns[cols],axis=1,inplace=True)

finalresultcopy.head(10)
finalresultcopy.to_csv("withdescription.csv")
# finalrulestest=pd.concat([finalrules,tags,tags2],axis=1,join='inner')
# =============================================================================



# need to get the value of frozenset 
# append department 
#remove duplicates
