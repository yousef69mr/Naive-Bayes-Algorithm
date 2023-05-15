import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score


def clean_dataset(dataset):

    # remove duplicates
    reduced_dataset = dataset.drop_duplicates()

    try:
        # replace missing data with zeros
        reduced_dataset.fillna(value=0, inplace=True)
        return reduced_dataset
    except Exception as e:
        # Log the error
        print(f"Error: {e}")
        return reduced_dataset.DataFrame()  # Return an empty DataFrame if an error occurs


def print_heatmap(numeric_dataset):
    corr = numeric_dataset.corr(method='pearson')
    cmap = sns.diverging_palette(250, 354, 80, 60, center='dark', as_cmap=True)
    sns.heatmap(corr, vmax=1, vmin=-.5, cmap=cmap, square=True, linewidths=.2)
    plt.show()


def print_histogram(numeric_dataset):
    columns = numeric_dataset.columns
    fig, axes = plt.subplots(1, len(columns), figsize=(18, 6), sharey='all')
    for i in range(len(columns)):
        sns.histplot(numeric_dataset, ax=axes[i], x=columns[i], kde=True)
    plt.show()


# calculate P(Y=y) for all possible y (output column)
# P(no) / P(yes)
def calculate_prior(dataframe, output_column):
    classes = sorted(list(dataframe[output_column].unique()))
    # print(classes)
    # print(dataframe[output_column])
    prior = []
    for i in classes:
        # print(len(dataframe[dataframe[output_column] == i]))
        prior.append(len(dataframe[dataframe[output_column] == i])/len(dataframe))

    return prior


# # calculate P(X=x|Y=y) using Gaussian distance
# def calculate_likelihood_gaussian(dataframe, feat_name, feat_val, y, label):
#     features = list(dataframe.columns)
#     dataframe = dataframe[dataframe[y] == label]
#     mean, std = dataframe[feat_name].mean(), dataframe[feat_name].std()
#     p_x_given_y = (1/(np.sqrt(2*np.pi)*std)) * np.exp(-((feat_val-mean)**2 / (2*std**2)))
#     return p_x_given_y
#

# calculate P(X=x|Y=y) categorically
def calculate_likelihood_categorical(dataframe, feature_name, feature_value, y, label):
    # features = list(dataframe.columns)
    dataframe = dataframe[dataframe[y] == label]
    # print(dataframe)
    p_x_given_y = len(dataframe[dataframe[feature_name] == feature_value]) / len(dataframe)
    # print(len(dataframe[dataframe[feature_name] == feature_value]))
    return p_x_given_y


#
# def naive_bayes_gaussian(dataframe, test_data, output_column):
#     # get features
#     features = list(dataframe.columns)[:-1]
#
#     # calculate prior
#     prior = calculate_prior(dataframe,output_column)
#
#     y_predict = []
#
#     for x in test_data:
#         labels = sorted(list(dataframe[output_column].unique()))
#         likelihood = [1]*len(labels)
#
#         for j in range(len(labels)):
#             for i in range(len(features)):
#                 likelihood[j] *= calculate_likelihood_gaussian(dataframe, features[i], x[i], output_column, labels[j])
#
#         # calculate posterior probability (numerator only)
#         post_prob = [1]*len(labels)
#         for i in range(len(labels)):
#             post_prob[i] = likelihood[i] * prior[i]
#
#         y_predict.append(np.argmax(post_prob))
#
#     return np.array(y_predict)
#


def naive_bayes_categorical(dataframe, test_data, output_column):
    # get features
    features = list(dataframe.columns)[:-1]
    # print(features)
    # calculate prior
    prior = calculate_prior(dataframe, output_column)
    # print(prior)
    y_predict = []

    labels = sorted(list(dataframe[output_column].unique()))  # ['no', 'yes']

    for x in test_data:

        # print(labels)
        likelihood = [1]*len(labels)  # [1, 1]
        # print(likelihood)

        # multiply of probabilities
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_categorical(dataframe, features[i], x[i], output_column, labels[j])

        # print(f'Likelihood : {likelihood}')
        # calculate posterior probability (numerator only)
        # multiply with prior probability
        post_prob = []
        for j in range(len(labels)):
            post_prob.append(likelihood[j] * prior[j])

        y_predict.append(np.argmax(post_prob))

    y_predict = [labels[predict] for predict in y_predict]
    return np.array(y_predict)


def run():
    dataset = pd.read_csv('Bank_dataset.csv', sep=";")
    # print(dataset)
    cleaned_dataset = clean_dataset(dataset)
    print(cleaned_dataset)
    output_column = cleaned_dataset.iloc[:, -1]
    # Select only the numeric columns
    numeric_dataset = cleaned_dataset.select_dtypes(include='number')
    # numeric_dataset = pd.concat([numeric_dataset, output_column], axis=1)
    # print(numeric_dataset)
    # numeric_data = np.array(numeric_dataset)

    print_heatmap(numeric_dataset)
    print_histogram(numeric_dataset)

    train, test = train_test_split(cleaned_dataset, test_size=.4, random_state=41)

    x_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values
    # print(len(y_test))
    y_predict = naive_bayes_categorical(train, x_test, output_column.name)

    # print(y_test)
    # print(y_predict)
    # print(len(y_predict))
    print(confusion_matrix(y_test, y_predict))
    print(f1_score(y_test, y_predict, pos_label='no'))


if __name__ == "__main__":
    run()
