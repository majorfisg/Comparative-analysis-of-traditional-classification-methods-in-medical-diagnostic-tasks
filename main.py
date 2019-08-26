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


# Разделяем на обучающую и тестовую выборки для первой части (до использования cross val score)
#df.iloc[:, :-1], df.iloc[:, -1] – правильно настроить столбец с метками классов
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], 
                                                    test_size = 0.5, random_state=1121, 
                                                    shuffle=True)

#Обычная оценка точности
#LogisticRegression
log = LogisticRegression()
log.fit(x_train, y_train)

acc_log_train = accuracy_score(y_train, log.predict(x_train))
acc_log_test = accuracy_score(y_test, log.predict(x_test))

#KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
acc_knn_train = accuracy_score(y_train, knn.predict(x_train))
acc_knn_test = accuracy_score(y_test, knn.predict(x_test))

#RandomForestClassifier
rfc = RandomForestClassifier(max_depth=2)
rfc.fit(x_train, y_train)
acc_rfc_train = accuracy_score(y_train, rfc.predict(x_train))
acc_rfc_test = accuracy_score(y_test, rfc.predict(x_test))

#GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
acc_gnb_train = accuracy_score(y_train, gnb.predict(x_train))
acc_gnb_test = accuracy_score(y_test, gnb.predict(x_test))
#VotingClassifier
ens = VotingClassifier([('LogR', LogisticRegression()), 
                        ('NaiveBayes',GaussianNB()), 
                        ('kNN', KNeighborsClassifier(n_neighbors=11)),
                        ('RandomForest', RandomForestClassifier(max_depth=2))], 
                        voting='soft', weights=[1, 1, 1, 1])
ens.fit(x_train, y_train)

acc_ens_train = accuracy_score(y_train, ens.predict(x_train))
acc_ens_test = accuracy_score(y_test, ens.predict(x_test))

#Строим столбчатую диаграмму для подсчитанных оценок

bar_width = 0.3 # Ширина столбца диаграммы
position = np.arange(5) # Указываем число позиций на диаграмме: 4 классификатора – 4 позиции

# Записываем в списки оценки точности, которые были получены выше
total_accuracy_train = [acc_log_train, acc_knn_train, 
                        acc_rfc_train, acc_gnb_train,
                        acc_ens_train] # Точность на обучении
total_accuracy_test = [acc_log_test, acc_knn_test, 
                       acc_rfc_test, acc_gnb_test,
                       acc_ens_test] # Точность на тесте

# Строим графики

plt.bar(position, total_accuracy_train, width = bar_width, 
        label='Обучающая выборка', align='center')
plt.bar(position+bar_width, total_accuracy_test, width=bar_width, 
        label='Тестовая выборка')
# Подписываем
plt.xticks(position+bar_width/2, ['LogR', 'KNN', 'RandomForest', 'GaussianNB', 'Ensemble'])
plt.ylabel('Точность')


plt.ylim((0, 1.1)) # Редактируем вертикальную ось
plt.legend(loc=1) # Указываем позицию легенды
plt.title('Точность классификации на обучающем и тестовом множестве'); # Печатаем заголовок


#Кросс-валидация
#Применяем кросс-валидацию: считаем оценки точности на количестве разбиений выборки num_of_folds, оцениваем дисперсию оценок
# Создаем список классификаторов
classifiers = [LogisticRegression(), GaussianNB(), 
               KNeighborsClassifier(), RandomForestClassifier(n_estimators=16, max_features=1), 
               VotingClassifier([('LogR', LogisticRegression()), 
                                 ('NaiveBayes',GaussianNB()), 
                                 ('kNN', KNeighborsClassifier(n_neighbors=11)),
                                 ('RandomForest', RandomForestClassifier(max_depth=2))], 
                                 voting='soft', 
                                 weights=[1, 2, 1, 1])]

# Создаем список имен для отображения на диаграммах
names = ['Logistic Regr', 'GaussianNB', 'KNN', 'RandomForest', 'Ensemble']

# Число блоков для кросс-валидации
num_of_folds = 100

###### Создаем словари для обучающей и тестовой выборок:
### Словарь, содержащий оценку точности на обуч. выборке для различных классификаторов  
cross_validation_acc = {} 
### Словарь, содержащий оценку дисперсии оценки точности на обуч. 
###выборке для различных классификаторов  
cross_validation_var = {} 



# В цикле по классификаторам проводим обучение, на кроссвалидации заполняем словари 
for i, clf in zip(names, classifiers):
    cross_validation_scores = cross_val_score(clf, df.iloc[:, :-1], 
                                              df.iloc[:, -1], cv=num_of_folds)
    # Средняя оценка точности на num_of_folds подвыборках
    cross_validation_acc[i] = cross_validation_scores.mean()
    
    # Средняя оценка дисперсии на num_of_folds подвыборках
    cross_validation_var[i] = cross_validation_scores.var()
    
    
    
pprint(cross_validation_acc)

#Сравнение точности кроссвалидации и обычного классификатора
plt.bar(np.arange(len(cross_validation_acc)), cross_validation_acc.values(), width=0.3, 
        yerr=np.std(list(cross_validation_acc.values())))
plt.bar(np.arange(len(total_accuracy_test))+0.3, total_accuracy_test, width=0.3,
        yerr=np.std(total_accuracy_test))
plt.xticks(np.arange(len(names))+0.3, names);

#Строим столбчатую диаграмму точности и дисперсии
plt.ylim((0, 1))
plt.xlim(-0.3, 4.3)
bw = 0.3

# Рисуем график
plt.bar(np.arange(len(cross_validation_acc)), cross_validation_acc.values(), width=bw)

# Чертим прямую линию по максимальной оценке точности
plt.plot([-1, 5], [max(cross_validation_acc.values()), 
                   max(cross_validation_acc.values())], c='red', alpha=0.4)

plt.xticks(np.arange(len(names)), names);
plt.title('Усредненная точность. Число разбиений: %i'%num_of_folds)

#Строим ROC-кривую, подсчитываем площадь под ней (AUC)
for i, j in zip(names, classifiers):
    j.fit(x_train, y_train)
    fpr, tpr, treshoulds = roc_curve(y_test, j.predict_proba(x_test)[:, 1])
    plt.plot(fpr, tpr, label='%s: auc=%0.2f'%(i, auc(fpr, tpr)))
    print('Method: %s, ' %i, 'auc: %.3f'%auc(fpr, tpr))

plt.plot([0, 1], [0, 1], '--', c='gray', alpha=0.2)

plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC-кривые для рассмотренных моделей')

plt.legend()
