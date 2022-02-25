import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report ,accuracy_score ,confusion_matrix
data =pd.read_csv('Breast_cancer_data.csv')
X=data.drop(['diagnosis'],axis=1)
y=data['diagnosis']
aver=data.groupby('diagnosis').size()
plt.figure(1)
aver.plot(kind='pie', title='Number of parties on diagnosis',labels=['0','1'],explode=[0.2,0],autopct='%.2f%%')
plt.legend(title='diagnosis statistics')
#plt.show()
plt.figure(2)
corr = X.corr()
sns.heatmap(corr,annot=True,cmap='Greens')
#plt.show()

#Modeling
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=.3,random_state=0)
#print(x_test.iloc[0])
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)
#print(x_test[0])

## Run SVM with default hyperparameters
# Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.
svm=SVC()
svm.fit(x_train,y_train)
y_pred_d=svm.predict(x_test)
#print(x_test)
print("the accuracy of SVM is = {:.2%}".format(accuracy_score(y_test,y_pred_d)))
print(classification_report(y_test,y_pred_d))
# #Linear SVM , C=1
# Linear_Svm=SVC(kernel='linear',C=1)
# Linear_Svm.fit(x_train,y_train)
# y_pred=Linear_Svm.predict(x_test)
# print("the accuracy of SVM is = {:.2%}".format(accuracy_score(y_test,y_pred)))
# print(classification_report(y_test,y_pred))

# Compare the train-set and test-set accuracy
y_pred_train = svm.predict(x_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))
# Check for overfitting and underfitting
print('Training set score: {:.4f}'.format(svm.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(svm.score(x_test, y_test)))
# #Polynomial SVM
# poly_svc100=SVC(kernel='poly', C=100.0)
# poly_svc100.fit(x_train, y_train)
# y_pred=poly_svc100.predict(x_test)
# print('Accuracy polynomial kernel and C=1.0 = {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_d)
print('Confusion matrix\n\n', cm)
print('\nTrue Positives(TP) = ', cm[0,0])
print('\nTrue Negatives(TN) = ', cm[1,1])
print('\nFalse Positives(FP) = ', cm[0,1])
print('\nFalse Negatives(FN) = ', cm[1,0])
# visualize confusion matrix with seaborn heatmap
plt.figure(3)
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()
# Stratified k-fold Cross Validation with shuffle split
kfold=KFold(n_splits=6,shuffle=True,random_state=0)
rbl_svm=SVC()
rbl_scores=cross_val_score(rbl_svm,x_train,y_train,cv=kfold)
# print cross-validation scores with linear kernel
print('Stratified cross-validation scores with linear kernel:\n\n{}'.format(rbl_scores))
# print average cross-validation score with linear kernel
print('Average stratified cross-validation score with linear kernel:{:.4f}'.format(rbl_scores.mean()))

#KNN
Knn=KNeighborsClassifier(n_neighbors=11)
Knn.fit(x_train,y_train)
y_pred=Knn.predict(x_test)
print("the accuracy of KNN is = {:.2%}".format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))

#Naive_Base
G_naive=GaussianNB()
G_naive.fit(x_train,y_train)
G_naive.predict(x_test)
y_pred=Knn.predict(x_test)
print("the accuracy of G_Naive_Base is = {:.2%}".format(accuracy_score(y_test,y_pred)))
print(classification_report(y_test,y_pred))
