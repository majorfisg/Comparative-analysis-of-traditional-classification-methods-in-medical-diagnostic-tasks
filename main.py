import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
%matplotlib inline
# plt.('figure', figsize=(10, 7))
plt.rcParams["figure.figsize"] = (10,7)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, classification_report, f1_score
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from scipy.stats import mode
from sklearn.model_selection import cross_val_score, LeaveOneOut
import random

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




import warnings
from pprint import pprint
warnings.filterwarnings('ignore')




# Загрузка heart
df = pd.read_csv('Heart_Disease_Data.csv', sep=',')
df.pred_attribute = df.pred_attribute.replace([1, 2, 3, 4], 1)
df = df.replace('?', 0, method='ffill')

# Загрузка liver (bupa)
# df = pd.read_csv('bupa.data.csv', sep=',')
# df.attribute = df.attribute.replace({2 : 1, 1 : 0})


# Загрузка diabets
# df = pd.read_csv('diabets.csv', sep=',')

df.head()