# -*- coding: utf-8 -*-
"""
Created on Sun May 31 14:19:13 2020

@author: renluqin
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import prince

"""load documents"""
combats = pd.read_csv("pokemon-challenge/combats.csv")
pokemon = pd.read_csv("pokemon-challenge/pokemon.csv")
tests = pd.read_csv("pokemon-challenge/tests.csv")
"""fill missing values"""
data = pokemon.copy()
data["Name"] = pokemon['Name'].fillna('Primeape')
data["Type 2"] = pokemon['Type 2'].fillna('No Type 2')
"""feature engineering"""
# calculate the win % of each pokemon 
# add the calculation to the pokemon dataset 
total_Wins = combats.Winner.value_counts()
# get the number of wins for each pokemon
numberOfWins = combats.groupby('Winner').count()
#both methods produce the same results
countByFirst = combats.groupby('Second_pokemon').count()
countBySecond = combats.groupby('First_pokemon').count()
numberOfWins = numberOfWins.sort_index()
numberOfWins['Total Fights'] = countByFirst.Winner + countBySecond.Winner
numberOfWins['Win Percentage']= numberOfWins.First_pokemon/numberOfWins['Total Fights']
# merge the winning dataset and the original pokemon dataset
results2 = pd.merge(data, numberOfWins, right_index = True, left_on='#')
results3 = pd.merge(data, numberOfWins, left_on='#', right_index = True, how='left')
results4 = results3.drop(columns=['First_pokemon', 'Second_pokemon','Total Fights'])
"""correlation map"""
f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(results4.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
"""PCA"""
results2["win"] = 'win percentage < 0.5'
results2.loc[results3["Win Percentage"]>0.5,"win"] = 'win percentage > 0.5'
X = results2.iloc[:,4:10]
pca = prince.PCA(n_components=6,
                 n_iter=3,
                 rescale_with_mean=True,
                 rescale_with_std=True,
                 copy=True,
                 check_input=True,
                 engine='auto',
                 random_state=42)
pca = pca.fit(X)
ax = pca.plot_row_coordinates(
     X,
     ax=None,
     figsize=(6, 6),
     x_component=0,
     y_component=1,
     labels=None,
     color_labels=results2.iloc[:,-1],
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True)
"""One-class SVM to detect outliers"""
from sklearn.svm import OneClassSVM
from numpy import quantile, where
svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.02)
pred = svm.fit_predict(X)
scores = svm.score_samples(X)
thresh = quantile(scores, 0.005)
index = where(scores<=thresh)
index = index[0]
values = X.iloc[index]
plt.scatter(X['Speed'], X['Attack'])
plt.scatter(values['Speed'], values['Attack'], color='r')
plt.xlabel('Speed')
plt.ylabel('Attack')
plt.title("Outliers according to one-class SVM", fontsize=18)

"""onehot+pca"""
from sklearn.preprocessing import  OneHotEncoder
results3[['Type 1','Type 2','Generation','Legendary']] = results3[['Type 1','Type 2','Generation','Legendary']].astype('category')
results3[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']] = results3[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].astype('float64')
quali = results3[['Type 1','Type 2', 'Generation', 'Legendary']]
enc = OneHotEncoder(categories = 'auto')
quali = enc.fit_transform(quali).toarray()
quali = pd.DataFrame(quali)
quant = results3[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
X = pd.concat( [quali, quant], axis=1 )
from sklearn.decomposition import PCA
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
cls = PCA(n_components=50)
pcs = cls.fit_transform(X)
df_X = pd.DataFrame(pcs, columns=[f"PC{i}" for i in range(1, 7)])
sns.scatterplot(x="PC1", y="PC2", hue = results2["win"],data=df_X)
plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5", "Axe 6"], cls.explained_variance_ratio_)
plt.title("Explained Variance Ratio", fontsize=20)







