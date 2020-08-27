# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:09:45 2020

@author: renluqin
"""

import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils import add_labels
import prince

combats = pd.read_csv("pokemon-challenge/combats.csv")
pokemon = pd.read_csv("pokemon-challenge/pokemon.csv")
tests = pd.read_csv("pokemon-challenge/tests.csv")

# pie plot
data = pokemon.copy()
data["Name"] = pokemon['Name'].fillna('Primeape')
type1_count = data['Type 1'].value_counts(dropna =False)  # if there are nan values that also be counted
# As it can be seen below there are 112 water pokemon or 70 grass pokemon
data["Type 2"] = pokemon['Type 2'].fillna('No Type 2')




sns.set(style="darkgrid")
ax = sns.countplot(y="Type 1", 
                   data=data,
                   order = data['Type 1'].value_counts().index)
ax.set_title('Répartition de Type 1', fontsize=18, position=(0.5,1.05))

ax = sns.countplot(y="Type 2", 
                   data=data,
                   order = data['Type 2'].value_counts(dropna =False).index)
ax.set_title('Répartition de Type 1', fontsize=18, position=(0.5,1.05))

ax = sns.countplot(y="Generation", 
                   data=data,
                   order = data['Generation'].value_counts(dropna =False).index)
ax.set_title('Répartition de Generation', fontsize=18, position=(0.5,1.05))

ax = sns.countplot(y="Legendary", 
                   data=data,
                   order = data['Legendary'].value_counts(dropna =False).index)
ax.set_title('Répartition de Legendary', fontsize=18, position=(0.5,1.05))



g = sns.FacetGrid(data, col="Type 1",col_wrap=6)
g.map(sns.distplot, "HP")

f, ax = plt.subplots(figsize = (14, 10))
sns.distplot(data["HP"].dropna(),ax=ax)
ax.set_title('Répartition de Hitpoints', fontsize=38, position=(0.5,1.05))


f, ax = plt.subplots(figsize = (14, 10))
sns.distplot(data["Attack"].dropna(),ax=ax)
ax.set_title('Répartition de Attack Force', fontsize=38, position=(0.5,1.05))

f, ax = plt.subplots(figsize = (14, 10))
sns.distplot(data["Defense"].dropna(),ax=ax)
ax.set_title('Répartition de Defense Points', fontsize=38, position=(0.5,1.05))


f, ax = plt.subplots(figsize = (14, 10))
sns.distplot(data["Sp. Atk"].dropna(),ax=ax)
ax.set_title('Répartition de Special Attack Force', fontsize=38, position=(0.5,1.05))

f, ax = plt.subplots(figsize = (14, 10))
sns.distplot(data["Sp. Def"].dropna(),ax=ax)
ax.set_title('Répartition de Special Defense Force', fontsize=38, position=(0.5,1.05))

f, ax = plt.subplots(figsize = (14, 10))
sns.distplot(data["Speed"].dropna(),ax=ax)
ax.set_title('Répartition de Speed', fontsize=38, position=(0.5,1.05))

g = sns.FacetGrid(data, hue="Legendary",palette='coolwarm')
g.map(sns.distplot, "HP")
g.fig.suptitle('Répartition de Hitpoints', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(data, hue="Legendary",palette='coolwarm')
g.map(sns.distplot, "Attack")
g.fig.suptitle('Répartition de Attack Force', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(data, hue="Legendary",palette='coolwarm')
g.map(sns.distplot, "Defense")
g.fig.suptitle('Répartition de Defense Points', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(data, hue="Legendary",palette='coolwarm')
g.map(sns.distplot, "Sp. Atk")
g.fig.suptitle('Répartition de Special Attack Force', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(data, hue="Legendary",palette='coolwarm')
g.map(sns.distplot, "Sp. Def")
g.fig.suptitle('Répartition de Special Defense Force', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(data, hue="Legendary",palette='coolwarm')
g.map(sns.distplot, "Speed")
g.fig.suptitle('Répartition de Speed', fontsize=14, position=(0.5,1.05)) 
g.add_legend()


"""
计算每一个generation的HP/Speed等平均值
"""
df_mean = pd.DataFrame(columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])

for i in range(1,7):
    df_mean = df_mean.append(np.mean(data[data['Generation']==i])[1:7], ignore_index=True)

df_mean["Legendary"] = [1,2,3,4,5,6]    

for i in range(6):
    plt.plot(df_mean.iloc[i],'-o',label = 'G%d'%(i+1))
plt.legend()
plt.title("Comparaison des moyennes de valeurs d'attributs de différentes générations")


#correlation map
f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()

g = sns.jointplot("Sp. Def", "Defense", data=data,size=5, ratio=3, color="r")
g.fig.suptitle('Corrélation entre la défense et la Sp défense', fontsize=14, position=(0.5,1.05)) 


g = sns.jointplot("Sp. Def", "Sp. Atk", data=data,size=5, ratio=3, color="r")
g.fig.suptitle('Corrélation entre la Sp défense et la Sp attaque', fontsize=14, position=(0.5,1.05)) 

g = sns.scatterplot(x="Sp. Def", y="Sp. Atk", hue="Legendary", size="Generation", data=data)
plt.title('La relation entre la génération, le légendaire, Sp défense et la Sp attaque', fontsize=14, position=(0.5,1.05)) 

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

results3.groupby('Type 1').agg({"Win Percentage": "mean"}).sort_values(by = "Win Percentage")

# We can look at the difference between the two datasets to see which pokemon never recorded a fight
#missing_Pokemon = np.setdiff1d(pokemon.index.values, results3.index.values)
#subset the dataframe where pokemon win percent is NaN
results3[results3['Win Percentage'].isnull()]

top10_bad_pokemon = results3[np.isfinite(results3['Win Percentage'])].sort_values(by = ['Win Percentage']).head(10)

g = sns.scatterplot(x="Sp. Def", y="Sp. Atk", hue="Legendary", size="Generation", data=top10_bad_pokemon)
plt.title('Les 10 pokemons les plus faibles', fontsize=14, position=(0.5,1.05)) 
add_labels(top10_bad_pokemon["Sp. Def"], top10_bad_pokemon["Sp. Atk"], top10_bad_pokemon["Name"])

top10_good_pokemon = results3[np.isfinite(results3['Win Percentage'])].sort_values(by = ['Win Percentage'], ascending = False ).head(10)
g = sns.scatterplot(x="Sp. Def", y="Sp. Atk", hue="Legendary", size="Generation", data=top10_good_pokemon)
plt.title('Les 10 pokemons les plus faibles', fontsize=14, position=(0.5,1.05)) 
add_labels(top10_good_pokemon["Sp. Def"], top10_good_pokemon["Sp. Atk"], top10_good_pokemon["Name"])


results4 = results3.drop(columns=['First_pokemon', 'Second_pokemon','Total Fights'])
#correlation map
f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(results4.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


sns.regplot(x="Speed", y="Win Percentage", data=results3, logistic=True).set_title("Speed vs Win Percentage")
ax = sns.regplot(x="Attack", y="Win Percentage", data=results3).set_title("Attack vs Win Percentage")

top_50_pokemon = results3[np.isfinite(results3['Win Percentage'])].sort_values(by = ['Win Percentage'], ascending = False ).head(392)

end_50_pokemon = results3[np.isfinite(results3['Win Percentage'])].sort_values(by = ['Win Percentage'] ).head(391)

results3["win"] = 0
results3.loc[results3["Win Percentage"]>0.5,"win"] = 1

g = sns.FacetGrid(results3, hue="win",palette='coolwarm')
g.map(sns.distplot, "HP")
g.fig.suptitle('Répartition de Hitpoints', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(results3, hue="win",palette='coolwarm')
g.map(sns.distplot, "Attack")
g.fig.suptitle('Répartition de Attack Force', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(results3, hue="win",palette='coolwarm')
g.map(sns.distplot, "Defense")
g.fig.suptitle('Répartition de Defense Points', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(results3, hue="win",palette='coolwarm')
g.map(sns.distplot, "Sp. Atk")
g.fig.suptitle('Répartition de Special Attack Force', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(results3, hue="win",palette='coolwarm')
g.map(sns.distplot, "Sp. Def")
g.fig.suptitle('Répartition de Special Defense Force', fontsize=14, position=(0.5,1.05)) 

g = sns.FacetGrid(results3, hue="win",palette='coolwarm')
g.map(sns.distplot, "Speed")
g.fig.suptitle('Répartition de Speed', fontsize=14, position=(0.5,1.05)) 

g.add_legend()



"""
计算每一个type的HP/Speed等平均值
"""
df_mean = pd.DataFrame(columns = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"])
labels = list(data["Type 1"].drop_duplicates())

for i in labels:
    df_mean = df_mean.append(np.mean(data[data['Type 1']==i])[1:7], ignore_index=True)

df_mean["Type 1"] = labels

for i in range(6):
    plt.plot(df_mean.iloc[i],'-o',label = 'G%d'%(i+1))
plt.legend()
plt.title("Comparaison des moyennes de valeurs d'attributs de différentes générations")



from matplotlib.font_manager import FontProperties

labels = np.array([u" HP ", u" Attack ", u" Defense ", u" Sp. Atk ", u" Sp. Def ", u" Speed "])

for i in range(len(df_mean)):
    stats = df_mean.iloc[i][:-1]
    
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats, [stats[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    
   
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    
    ax.set_thetagrids(angles * 180/np.pi, labels)
    plt.title(df_mean.iloc[i][-1],fontsize=20)
plt.show()



type_cross = pd.crosstab(data["Type 1"], data["Type 2"])
type_cross.plot.bar(stacked=True, figsize=(14,4))
plt.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', ncol=5, fontsize=8, title="Type 2")
plt.title("Distribution conjointe de Type1 et Type2", fontsize=24)
plt.show()


"""
PCA
"""
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
     show_points=True
 )
pcs = pca.explained_inertia_ 
#array([0.46018404, 0.17834976, 0.12610354, 0.11881374, 0.0708509 ,0.04569803])
four_pcs_sum = sum(pcs[:4])

from sklearn.decomposition import PCA
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
cls = PCA(n_components=6)
pcs = cls.fit_transform(X)
df_X = pd.DataFrame(pcs, columns=[f"PC{i}" for i in range(1, 7)])
sns.scatterplot(x="PC1", y="PC2", hue = results2["win"],data=df_X)




plt.bar(["Axe 1", "Axe 2", "Axe 3", "Axe 4", "Axe 5", "Axe 6"], cls.explained_variance_ratio_)
plt.title("Explained Variance Ratio", fontsize=20)

"""
MCA
"""
X = results2.iloc[:,[2,3,10,11]]
X[['Type 1','Type 2','Generation','Legendary']] = X[['Type 1','Type 2','Generation','Legendary']].astype('category')


mca = prince.MCA(
     n_components=100,
     n_iter=3,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42
 )
mca = mca.fit(X)

ax = mca.plot_coordinates(
     X=X,
     ax=None,
     figsize=(6, 6),
     show_row_points=True,
     row_points_size=10,
     show_row_labels=False,
     show_column_points=True,
     column_points_size=30,
     show_column_labels=False,
     legend_n_cols=1
 )

len(mca.eigenvalues_)
mca.total_inertia_
mca.explained_inertia_
xlabels = ['{}'.format(t) for t in range(1,16)]
plt.bar(xlabels, mca.explained_inertia_[:15])
plt.xlabel("Axe")
plt.title("Explained Inertia Ratio", fontsize=20)

"""
FAMD
"""
X = results2.iloc[:,2:12]
X[['Type 1','Type 2','Generation','Legendary']] = X[['Type 1','Type 2','Generation','Legendary']].astype('category')

famd = prince.FAMD(
     n_components=10,
     n_iter=3,
     copy=True,
     check_input=True,
     engine='auto',
     random_state=42
 )
famd = famd.fit(X)

famd.row_coordinates(X)

ax = famd.plot_row_coordinates(
     X,
     ax=None,
     figsize=(6, 6),
     x_component=0,
     y_component=1,
     color_labels=['{}'.format(t) for t in results2["win"]],
     ellipse_outline=False,
     ellipse_fill=True,
     show_points=True
 )


from sklearn.preprocessing import  OneHotEncoder

results3[['Type 1','Type 2','Generation','Legendary']] = results3[['Type 1','Type 2','Generation','Legendary']].astype('category')
results3[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']] = results3[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']].astype('float64')



quali = results3[['Type 1','Type 2', 'Generation', 'Legendary']]
enc = OneHotEncoder(categories = 'auto')
quali = enc.fit_transform(quali).toarray()
quali = pd.DataFrame(quali)

quant = results3[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]

X = pd.concat( [quali, quant], axis=1 )

"""
One-class SVM to detect outliers
"""
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
from numpy import quantile, where, random


svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.03)
print(svm)

svm.fit(X)
pred = svm.predict(X)

anom_index = where(pred==-1)
index = anom_index[0]
values = X.iloc[index]
plt.scatter(X['Attack'], X['Defense'])
plt.scatter(values['Attack'], values['Defense'], color='r')

svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.02)
print(svm)

pred = svm.fit_predict(X)
scores = svm.score_samples(X)

thresh = quantile(scores, 0.005)
print(thresh)

index = where(scores<=thresh)
index = index[0]
values = X.iloc[index]

plt.scatter(X['Speed'], X['Attack'])
plt.scatter(values['Speed'], values['Attack'], color='r')
plt.xlabel('Speed')
plt.ylabel('Attack')
plt.title("Outliers according to one-class SVM", fontsize=18)

#315 331 584 687





















