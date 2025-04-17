from matplotlib import pyplot as plt
from sklearn import linear_model
import re
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.models import Word2Vec
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import normalize
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
import random


def simple_decision_tree(X_train, y_train, X_test, y_test):
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    score = regressor.score(X_test, y_test)

    print("R^2 score:")
    print(score)


def simple_linear_regression(X_train, y_train, X_test, y_test):
    # Train model
    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)

    # Evaluate model
    score = clf.score(X_test, y_test)
    print("R^2 score:")
    print(score)

    coef = clf.coef_
    print("Coefficients:")
    print(coef)

    intercept = clf.intercept_
    print("Intercept:")
    print(intercept)

    return


def simple_poisson_regression(X_train, y_train, X_test, y_test):
    # Train model
    clf = linear_model.PoissonRegressor()
    clf.fit(X_train, y_train)

    # Evaluate model
    score = clf.score(X_test, y_test)
    print("R^2 score:")
    print(score)

    print(y_test[:10])
    for i in range(10):
        print(clf.predict(X_test[i].reshape(1, -1)))

    return


def vectorize_and_clean_strings(titles: list[str], vector_size: int = 100):
    # Function to clean text
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = " ".join(
            [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
        )  # Remove stopwords
        return text

    # Apply cleaning to the titles
    cleaned_titles = [clean_text(title) for title in titles]

    # Tokenize the cleaned titles
    tokenized_titles = [title.split() for title in cleaned_titles]

    # Train a Word2Vec model
    word2vec_model = Word2Vec(
        sentences=tokenized_titles,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        seed=42,
    )

    # Create dense vectors for each title by averaging word vectors
    vectorized_words = np.array(
        [
            (
                np.mean(
                    [
                        word2vec_model.wv[word]
                        for word in title
                        if word in word2vec_model.wv
                    ],
                    axis=0,
                )
                if any(word in word2vec_model.wv for word in title)
                else np.zeros(vector_size)
            )
            for title in tokenized_titles
        ]
    )

    # Normalize the vectors
    vectorized_words = normalize(vectorized_words)

    return vectorized_words


def remove_duplicates(df, col: str):
    temp = df.clone()
    unique = temp.clear()

    for url in df[col].unique():
        dup = df.filter(pl.col(col) == url)
        if dup.shape[0] > 1:
            unique.vstack(dup, in_place=True)

    return unique


def undersample(df, limit: int = 10):

    # Undersample low scores
    high_score = df.filter(pl.col("score") >= limit)
    low_score = df.filter(pl.col("score") < limit)
    low_score_sampled = low_score.sample(n=len(high_score))

    df = pl.concat([low_score_sampled, high_score])
    return df


def scale_data(X, y):
    y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()
    X = StandardScaler().fit_transform(X)

    return X, y


def parameter_tuning(rf, y_train, X_train):
    param_grid = {
        "n_estimators": [100, 400, 900],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 8, 10],
        "max_features": ["sqrt", "log2"],
    }
    cv = RandomizedSearchCV(rf, param_distributions=param_grid)
    cv.fit(X_train, y_train)
    print("Best parameters found: ", cv.best_params_)
    print("Best cross-validation score: ", cv.best_score_)
    return


def get_decision_rules_from_forest(rf, X_test, y_test, feature_names):
    # Get the predictions from each tree
    y_preds = np.zeros((len(X_test), rf.n_estimators))
    for i, tree in enumerate(rf.estimators_):
        y_preds[:, i] = tree.predict(X_test)

    # Compute the Mean Squared Error for each tree
    errors = np.mean((y_preds - y_test[:, np.newaxis]) ** 2, axis=0)

    # Find the index of the tree closest to the mean error
    mean_error = np.mean(errors)
    closest_tree_idx = np.argmin(np.abs(errors - mean_error))

    # Get the most successful tree (closest to the mean)
    best_tree = rf.estimators_[closest_tree_idx]

    # Plot the most successful tree
    plt.figure(figsize=(12, 12))
    plot_tree(best_tree, feature_names=feature_names, fontsize=10, max_depth=4)
    plt.show()

    # Export the decision rules of the best tree
    tree_rules = export_text(best_tree, feature_names=feature_names)

    # Print the rules of the most successful tree
    print("Decision rules of the most successful tree:")
    print(tree_rules)

    return


def mean_vs_median(rf, X_train, y_train, X_test, y_test):
    rf.fit(X_train, y_train)

    print("Ground truth:")
    # Randomly select 10 indices from the test set
    indices = random.sample(range(len(y_test)), 10)
    print(indices)
    print("--")
    print(y_test[indices])

    dict = {x: [] for x in range(10)}
    for tree in range(100):
        pred = rf.estimators_[tree].predict(X_test[indices])
        for i, idx in enumerate(indices):
            dict[i].append(pred[i])

    res_mead = []
    res_mean = []
    for x, y in dict.items():
        y.sort()
        res_mead.append(int(y[50]))
        res_mean.append(int(sum(y) / len(y)))

    print("Mean and median of tree predictions")
    print(res_mean)
    print(res_mead)

    error_mead = []
    error_mean = []
    for i in indices:
        error_mead.append(abs(y_test[i] - res_mead[i]))
        error_mean.append(abs(y_test[i] - res_mean[i]))

    print("Total absolute errors")
    print(error_mean)
    print(error_mead)

    print("Mean error")
    print(sum(error_mean) / len(error_mean))
    print(sum(error_mead) / len(error_mead))


def in_out_sample(rf, X_train, y_train, X_test, y_test):
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)
    if hasattr(rf, "predict_proba"):  # Check if the random forest is a classifier
        print("In-sample prediction:", np.mean(y_train == y_train_pred))
        print("Out-of-sample prediction:", np.mean(y_test == y_test_pred))
    else:  # Otherwise is regressor
        print("In-sample prediction:", np.sqrt(np.mean((y_train - y_train_pred) ** 2)))
        print(
            "Out-of-sample prediction:", np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        )


# %%
