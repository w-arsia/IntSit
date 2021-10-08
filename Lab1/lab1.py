import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:\\Users\\polina\\Downloads\\Filialy.dat', sep=";", encoding='utf-8', skipinitialspace=True)
df
df.info()

df.rename(columns={'НАЗВАНИЕ':'name','ПЛОЩАДЬ':'area','ПРОХОДИМ':'passability','АССОРТИМ':'assortment','КОНКУРЕН':'competitor','МЕТРО':'metro','КОНСУЛЬТ':'consultant','ДИЗАЙН':'design','ЦЕНЫ':'price','ПРОДАЖИ':'sales'}, inplace=True)
df

df.drop('name',axis=1,inplace=True)
df

df['assortment'] = df['assortment'].apply(lambda x: x.strip())
df['competitor'] = df['competitor'].apply(lambda x: x.strip())
df['consultant'] = df['consultant'].apply(lambda x: x.strip())
df['design'] = df['design'].apply(lambda x: x.strip())

df['assortment'].unique()

df['competitor'].unique()

df['consultant'].unique()

df['design'].unique()

assortmentDict = {'миним': 0, 'средний': 1, 'широкий': 2, 'макс': 3}
df['assortment'] = df['assortment'].map(assortmentDict)
competitorDict = {'хуже': -1, 'одинак': 0, 'лучше': 1}
df['competitor'] = df['competitor'].map(competitorDict)
consultantDict = {'Нет': 0, 'Есть': 1}
df['consultant'] = df['consultant'].map(consultantDict)
designDict = {'Вывеска': 0, 'Витрина': 1, 'Св+Ви': 2, 'Световая': 3, 'Бедно': 4, 'Вы+Ви': 5}
df['design'] = df['design'].map(designDict)

dfForX = df.copy()
dfForX.drop('sales',axis=1,inplace=True)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
def graph():

    m = dfForX.mean(axis=1)
    for i, col in enumerate(dfForX):
        dfForX.iloc[:, i] = dfForX.iloc[:, i].fillna(m)

    X = dfForX.iloc[:, 1: -1].values
    L = np.array(dfForX.iloc[:, :-1])
    print(L)
    L1 = []
    for i in L:
        L1.append(i[0])
    linked = linkage(X, method='average', metric='euclidean')
    plt.figure(figsize=(10, 10))
    dendrogram(linked, labels=L1)
    plt.axhline(70, color='r')  # 1.823
    plt.show()
    label = fcluster(linked, 70, criterion='distance')
    np.unique(label)
    dfForX.loc[:, 'label'] = label


# функция построения дендрограммы
graph()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = dfForX.values[:,:]
X = np.nan_to_num(X)
clust_data = StandardScaler().fit_transform(X)

SSE = []
for k in range(1,10):
    estimator = KMeans (n_clusters = k)
    estimator.fit(X)
    SSE.append(estimator.inertia_)
r = range(1,10)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(r,SSE,'o-')

#метод локтя
plt.show()

cluster_number = 3
k_means = KMeans(init = "k-means++", n_clusters = cluster_number, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
labels

df["cluster"] = labels
df

import matplotlib.pyplot as plt

for i in range(cluster_number):
  df.loc[df['cluster'] == i]
  plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['area'], alpha=0.65, label = i)

# plt.scatter(df['sales'],df['area'], c=labels.astype(np.float), alpha=0.6)
plt.xlabel('Sales')
plt.ylabel('area')
plt.legend()
plt.show()

# import matplotlib.pyplot as plt
#
# for i in range(cluster_number):
#   df.loc[df['cluster'] == i]
#   plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['passability'], alpha=0.65, label = i)
#
# plt.xlabel('Sales')
# plt.ylabel('passability')
# plt.legend()
# plt.show()

# import matplotlib.pyplot as plt
# for i in range(cluster_number):
#   df.loc[df['cluster'] == i]
#   plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['assortment'], alpha=0.65, label = i)
# plt.xlabel('Sales')
# plt.ylabel('assortment')
# plt.legend()
# plt.show()
#
# import matplotlib.pyplot as plt
# for i in range(cluster_number):
#   df.loc[df['cluster'] == i]
#   plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['competitor'], alpha=0.65, label = i)
# plt.xlabel('Sales')
# plt.ylabel('competitor')
# plt.legend()
# plt.show()
#
# import matplotlib.pyplot as plt
# for i in range(cluster_number):
#   df.loc[df['cluster'] == i]
#   plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['metro'], alpha=0.65, label = i)
# plt.xlabel('Sales')
# plt.ylabel('metro')
# plt.legend()
# plt.show()
#
# import matplotlib.pyplot as plt
# for i in range(cluster_number):
#   df.loc[df['cluster'] == i]
#   plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['consultant'], alpha=0.65, label = i)
# plt.xlabel('Sales')
# plt.ylabel('consultant')
# plt.legend()
# plt.show()
#
# import matplotlib.pyplot as plt
# for i in range(cluster_number):
#   df.loc[df['cluster'] == i]
#   plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['design'], alpha=0.65, label = i)
# plt.xlabel('Sales')
# plt.ylabel('design')
# plt.legend()
# plt.show()
#
# import matplotlib.pyplot as plt
# for i in range(cluster_number):
#   df.loc[df['cluster'] == i]
#   plt.scatter(df.loc[df['cluster'] == i]['sales'],df.loc[df['cluster'] == i]['price'], alpha=0.65, label = i)
# plt.xlabel('Sales')
# plt.ylabel('price')
# plt.legend()
# plt.show()




