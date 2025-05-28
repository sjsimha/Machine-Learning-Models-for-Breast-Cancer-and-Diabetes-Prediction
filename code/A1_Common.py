import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler


def get_scoring():
    return 'average_precision'


def get_preprocessed_train_test_data(dataset):
    if dataset == 'Pima':
        f_name = 'pima_diabetes.csv'
    else:
        f_name = 'breast-cancer-wisconsin.csv'

    df = pd.read_csv(filepath_or_buffer=f'data/{f_name}', sep=',', header=0)
    df.drop(df.columns[0], axis=1, inplace=True)
    x_ = df.iloc[:, 0:-1].values
    y_ = df.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2, stratify=y_, shuffle=True,
                                                        random_state=12)
    robust_scaler = RobustScaler()
    stdscaler = StandardScaler()
    x_train_scaled = stdscaler.fit_transform(x_train)
    x_test_scaled = stdscaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test


def plot_two_series(x, y1, y2, title, x_label, y_label, y1_legend, y2_legend, file_name, save_to_file=True,
                    show_plot=False, annotation_text='', drawstyle='default', xscale=None):
    fig, ax = plt.subplots()
    ax.plot(x, y1, label=y1_legend, marker="o", drawstyle=drawstyle)
    ax.plot(x, y2, label=y2_legend, marker="*", drawstyle=drawstyle)
    plt.title(title, fontsize=10)
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)

    if xscale is not None:
        ax.set_xscale(xscale)

    ax.text(0.5, 0.5, annotation_text, transform=ax.transAxes,
            fontsize=14, color='gray', alpha=0.5,
            ha='center', va='center', rotation=0)
    plt.legend()
    plt.grid()

    if save_to_file:
        plt.savefig(f'results/{file_name}', format='png')

    if show_plot:
        plt.show()

    plt.close()


def plot_learning_curve(clf, x_train, y_train, title, img_name, annotation):
    train_sizes, train_scores, test_scores = learning_curve(clf, x_train, y_train, scoring=get_scoring(), cv=5,
                                                            train_sizes=np.linspace(0.1, 1.0, 20), shuffle=True)

    plot_two_series(train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1),
                    title, '# of Samples', 'Average Precision', 'Training Score',
                    'Validation Score', img_name, save_to_file=True, show_plot=False,
                    annotation_text=annotation)


def plot_validation_curve(clf, x_train, y_train, param_name, param_range, x_axis_range, title, file_name, x_label,
                          results_file, annotation_text='', drawstyle='default', xscale=None):
    train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name=param_name, cv=5,
                                                 scoring=get_scoring(),
                                                 param_range=param_range)

    x_ = np.asarray(x_axis_range).reshape(len(x_axis_range), 1)
    y1_ = np.mean(train_scores, axis=1)
    y1_ = y1_.reshape(len(y1_), 1)
    y2_ = np.mean(test_scores, axis=1)
    y2_ = y2_.reshape(len(y2_), 1)

    with open(f'results/{results_file}', 'a') as sf:
        sf.write(f'\n****Validation Curve Results {title} {annotation_text} '
                 f'\n{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        results_ = np.concatenate((x_, y1_, y2_), axis=1)
        df = pd.DataFrame(results_)
        df.to_csv(sf)

    plot_two_series(x_axis_range, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1),
                    title, x_label, 'Average Precision', 'Training Score',
                    'Cross Validation Score', file_name, save_to_file=True, show_plot=False,
                    annotation_text=annotation_text,
                    drawstyle=drawstyle, xscale=xscale)


def perform_grid_search(clf, param_grid, x_train, y_train, cv=5, verbose=1, write_results=False, file_name=''):
    gs = GridSearchCV(clf, param_grid, scoring=get_scoring(), cv=cv, verbose=verbose, n_jobs=-1)
    gs.fit(x_train, y_train)

    results = pd.DataFrame(gs.cv_results_)[['params', 'mean_test_score', 'rank_test_score']]
    results.sort_values('rank_test_score', inplace=True)

    if write_results:
        with open(f'results/{file_name}', 'w') as sf:
            sf.write(f'\n{param_grid}\n\n')
            results.to_csv(sf)

    return gs


def plot_prc_curve(clf, x_test, y_test, title, img_name):
    metrics.PrecisionRecallDisplay.from_estimator(clf, x_test, y_test)
    ax = plt.subplot()

    ax.set_title(title)
    plt.savefig(f'results/{img_name}', format='jpg')
    plt.close()


def write_model_performance(clf, x_train, y_train, x_test, y_test, file_name):
    y_pred = clf.predict(x_test)
    probs_test = clf.predict_proba(x_test)[:, 1]

    with open(f'results/{file_name}', 'w') as sf:
        sf.write(f'{metrics.classification_report(y_test, y_pred)}\n\n')
        sf.write(f'{metrics.confusion_matrix(y_test, y_pred)}\n\n')
        sf.write(f'{metrics.average_precision_score(y_test, probs_test)}\n\n')
        sf.write(f'\nTraining Time: {get_training_time(clf, x_train, y_train)}s')
        sf.write(f'\nPrediction Time: {get_prediction_time(clf, x_test)}s\n')


def write_training_time(start_time, end_time):
    print(f'Model Learning Wall Clock Time: {(end_time - start_time):.2f}s ')


def get_prediction_time(clf, x_test):
    start_time = time.time()
    clf.predict(x_test)
    end_time = time.time()

    return calc_wall_clock_duration(start_time, end_time)


def get_training_time(clf, x_train, y_train):
    start_time = time.time()
    clf.fit(x_train, y_train)
    end_time = time.time()

    return calc_wall_clock_duration(start_time, end_time)


def calc_wall_clock_duration(start_time, end_time):
    return round(end_time - start_time, 4)
