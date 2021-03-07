# Kaggle Titanic Competition
# https://www.kaggle.com/c/titanic/overview
# Tried RandomForest (77.5%)

import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.model_selection import cross_val_score

def getCabinBlock(cabin):
    if cabin and isinstance(cabin, str):
        return str(cabin)[0]
    else:
        return 'X'

# Visualize relationship between 2 classes
def visualizeRelationship(dataset, xfeature, yfeature, title):
    xresult = []
    unique_vals = dataset[yfeature].unique()
    unique_vals.sort()
    for val in unique_vals:
        xresult.append(dataset[dataset[yfeature] == val][xfeature].value_counts())
    df_class = pd.DataFrame(xresult)
    df_class.index = [str(x) for x in unique_vals]
    df_class.plot(kind='bar', stacked=True, figsize=(5, 3), title = title)
    pyplot.show()

def runCrossValidation(model, X, Y, X_test):
    scores = cross_val_score(model, X, Y, cv=5)
    print(scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

def RunRandomForest(X, Y, X_test):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
    model.fit(X, Y)
    print("Training score: ", model.score(X,Y) * 100, "%")
    Y_test = model.predict(X_test)
    return Y_test, 'random_forest'

def RunKNearestNeighbour(X, Y, X_test):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree')
    model.fit(X, Y)
    print("Training score: ", model.score(X, Y) * 100, "%")
    runCrossValidation(model, X,Y, X_test)
    Y_test = model.predict(X_test)
    return Y_test, 'knn'

def RunLinearRegression(X, Y, X_test):
    from sklearn.linear_model import LinearRegression, Lasso
    # model = LinearRegression()
    alphas = np.logspace(-4, -1, 6)
    model = Lasso().set_params(alpha=alphas)
    scores = [model.set_params(alpha=alpha).fit(X, Y).score(X, Y)
                  for alpha in alphas]
    best_alpha = alphas[scores.index(max(scores))]
    model.alpha = best_alpha
    model.fit(X,Y)
    print("Training score: ", model.score(X, Y) * 100, "%")
    runCrossValidation(model, X,Y, X_test)
    Y_test = model.predict(X_test)
    return Y_test, 'linear_regression'

def RunSVM(X, Y, X_test):
    from sklearn.svm import SVC
    model = SVC(kernel='linear')
    model.fit(X, Y)
    print("Training score: ", model.score(X, Y) * 100, "%")
    runCrossValidation(model, X,Y, X_test)
    Y_test = model.predict(X_test)
    return Y_test, 'svm'

def RunNN(X, Y, X_test):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,10), random_state=1)
    model.fit(X, Y)
    print("Training score: ", model.score(X, Y) * 100, "%")
    runCrossValidation(model, X,Y, X_test)
    Y_test = model.predict(X_test)
    return Y_test, 'nn'

if __name__=="__main__":
    # Load data
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    passengerIds = test_data['PassengerId']

    # Pre-processing
    ## Apply transformation on columns
    train_data['Cabin'] = train_data['Cabin'].apply(getCabinBlock)
    test_data['Cabin'] = test_data['Cabin'].apply(getCabinBlock)

    # Visualize relations
    # visualizeRelationship(train_data, 'Cabin', 'Pclass', "Cabin by Class")

    ## Drop unwanted columns
    # Dropping Cabin to avoid Curse of Dimensionality
    Y = train_data["Survived"]
    train_data.drop(columns=['Survived','PassengerId','Name','Ticket','Fare','Cabin'], inplace=True)
    test_data.drop(columns=['PassengerId','Name','Ticket','Fare','Cabin'], inplace=True)
    # train_data.to_csv('train_mod.csv', index=False)

    ## Handle empty data
    train_data.fillna(0, inplace=True)
    test_data.fillna(0, inplace=True)

    ## Transform categorical features to binary features
    X = pd.get_dummies(train_data, columns=['Pclass','Sex','Embarked'])
    X_test = pd.get_dummies(test_data, columns=['Pclass','Sex','Embarked'])

    ## Handle missing columns in test data after above transformation
    missing_cols = set(X.columns) - set(X_test.columns)
    for c in missing_cols:
        X_test[c] = 0
    ### Ensure the order of column in the test set is in the same order than in train set
    X_test = X_test[X.columns]

    # Train Model and generate predictions
    Y_test, name = RunNN(X, Y, X_test)

    # Save
    output = pd.DataFrame({'PassengerId': passengerIds, 'Survived': Y_test})
    output.to_csv(name+'_out.csv', index=False)
    # print(predictions)

