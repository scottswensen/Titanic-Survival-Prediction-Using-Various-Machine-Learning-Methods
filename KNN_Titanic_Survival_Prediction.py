# Prediction of survival on the Titanic using K Nearest Neighbors (KNN)
"""In this project, the data from Kaggle's Titanic survival data challenge is 
analyzed using different machine learning methods to determine which method
best predicts the survival rate observed in the data. The predictors include
passenger class, age, number of siblings and/or spouses on board, number of 
parents and/or children on board, the ticket fare, passenger sex, and the port
of embarkation for each passenger. The data set is available here: 
https://www.kaggle.com/c/titanic
The training set was split and used for training and validation since the
provided test set on the Kaggle site does not include survival outcome.
"""
# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import the data set
df = pd.read_csv('train.csv')

# Set the index column as PassengerId
df.set_index('PassengerId', inplace = True)

# Drop the 'Cabin' column since most instances do not have information in this column
df.drop('Cabin', axis=1, inplace = True)

# Fill in missing age data using mean age by class
age_c1 = df[df['Pclass'] == 1]['Age'].mean()
age_c2 = df[df['Pclass'] == 2]['Age'].mean()
age_c3 = df[df['Pclass'] == 3]['Age'].mean()
def age_calc(cols):
    """ This function fills in missing ages with the mean age of that passenger's class"""
    age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(age):
        if Pclass == 1:
            return age_c1
        elif Pclass == 2:
            return age_c2
        else:
            return age_c3
    else:
        return age

df['Age'] = df[['Age','Pclass']].apply(age_calc, axis=1)    

# Only two entries have missing embarkaction information. Drop these instances.
df.dropna(inplace = True)    

# Import data test categorical columns - sex and embarkaction location
sex = pd.get_dummies(df['Sex'], drop_first = True)
embark = pd.get_dummies(df['Embarked'], drop_first=True)

df.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
df = pd.concat([df, sex, embark], axis=1)

# Add column value for family size, which is equal to 1 plus Parch + SibSp
def famsize(cols):
    sibSp = cols[0]
    parch = cols[1]
    return (1 + sibSp + parch)

df['FamSize'] = df[['SibSp', 'Parch']].apply(famsize, axis=1)

# Add column value that indicates if passenger is traveling alone
def solotrav(col):
    famsize = col[0]
    if famsize == 1:
        return 1
    else:
        return 0
    
df['Alone'] = df[['FamSize']].apply(solotrav, axis = 1)

# Visualize survival data by gender - women more likely to survive
plt.figure()
sns.countplot(x='Survived', hue='male', data=df, palette = 'Set1')
plt.title('Survival Rate by Gender')
plt.show()

# Visualize survival data by class - third class most likely to die
plt.figure()
sns.countplot(x='Survived', hue='Pclass', data=df, palette = 'Set1')
plt.title('Survival Rate by Passenger Class')
plt.show()

# Visualize survival data by fare - lower fare passengers more likely to die
plt.figure()
fig, ax = plt.subplots()
binsize = 30
ax.hist([df[df['Survived'] == 1]['Fare'], df[df['Survived'] == 0]['Fare']], binsize, histtype='bar', 
        align='mid', label=['Survived','Died'], alpha=0.5)
ax.set_xlabel('Fare [USD]')
ax.set_ylabel('Count')
ax.legend()
plt.title('Survival Rate by Fare Paid')
plt.show()

# Visualize survival data by age - most who died were 20 to 30 years old
plt.figure()
fig, ax = plt.subplots()
binsize = 30
ax.hist([df[df['Survived'] == 1]['Age'], df[df['Survived'] == 0]['Age']], binsize, histtype='bar', 
        align='mid', label=['Survived','Died'], alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.legend()
plt.title('Survival Rate by Age')
plt.show()

# Visualize survival data by number of siblings/spouses aboard
plt.figure()
sns.countplot(x='SibSp', hue='Survived', data=df, palette = 'RdBu')
plt.legend(labels = ['Died', 'Survived'], loc='upper right')
plt.title('Survival Rate by Number of Siblings/Spouses on Board')
plt.show()

# Visualize survival data by number of parents/children aboard
plt.figure()
sns.countplot(x='Parch', hue='Survived', data=df, palette = 'RdBu')
plt.legend(labels = ['Died', 'Survived'], loc='upper right')
plt.title('Survival Rate by Number of Parents/Children on Board')
plt.show()

# Visualize survival data by travelers traveling alone v with family members - solo travelers more likely to die
plt.figure()
sns.countplot(x='Alone', hue='Survived', data=df, palette = 'RdBu')
plt.legend(labels = ['Died', 'Survived'], loc='upper right')
plt.title('Survival Rate for Solo Travelers and Those Traveling with Family')
plt.show()

# Split the data into training and test sets. While the data provided by Kaggle
# includes a training and test set, the survival outcome is only provided for 
# the training set. Thus, to validate the model, the provided training set will
# be split into a training and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare','male', 'Q','S', 'FamSize', 'Alone']], df['Survived'], test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = pd.DataFrame(sc.fit_transform(X_train))
X_test_sc = pd.DataFrame(sc.transform(X_test))

# Fit the K-Nearest Neighbors model to the data
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors = 18)
model_knn.fit(X_train_sc, y_train)
pred_knn = model_knn.predict(X_test_sc)

# Choose an appropriate K Value by iterating through many values
error = []
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_sc,y_train)
    pred_i = knn.predict(X_test_sc)
    error.append(np.mean(pred_i != y_test))
plt.figure()
plt.plot(range(1,100),error,color='red')
plt.title('Error v. K Value')
plt.xlabel('K')
plt.ylabel('Error')

# Show confusion matrix and classification report
from sklearn.metrics import confusion_matrix, classification_report
print('Confusion Matrix: \n', confusion_matrix(y_test, pred_knn), '\n')
print('Classification Report: \n', classification_report(y_test, pred_knn), '\n')