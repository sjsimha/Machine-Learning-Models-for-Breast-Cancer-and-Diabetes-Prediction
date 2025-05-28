import numpy as np
from sklearn import svm

import A1_Common as a1c
import Svm_Common as sc


def run():
    # Step 1. Split into training and testing datasets
    x_train, x_test, y_train, y_test = a1c.get_preprocessed_train_test_data(dataset)

    # Step 2. Generate learning curve for base model;
    clf = svm.SVC(random_state=12)

    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_untuned'],
                            img_names_['learn_curve_untuned'], sc.make_annotation_string(clf))
    #   Observations
    #       Base model exhibits low bias and low Variance
    #           Training Average Precision score ~0.99
    #           Validation Set Average Precision score ~0.98
    #       However learning curves show decreasing validation performance as sample increase
    #

    # Step 3. Validation curve to determine impact of kernel (Hyperparameter kernel)
    param_range = ['linear', 'poly', 'rbf', 'sigmoid']
    a1c.plot_validation_curve(clf, x_train, y_train, 'kernel', param_range, param_range,
                              plot_titles_['val_curve_kernel'], img_names_['val_curve_kernel'],
                              'Kernel Type',
                              results_file=results_file_,
                              annotation_text=sc.make_annotation_string(clf, 'kernel'))
    #       Observations
    #           * Polynomial seemed the best choice
    clf.kernel = 'poly'
    clf.coef0 = 1

    # Validation curve for polynomial degree
    param_range = [1, 2, 3, 4, 5]
    a1c.plot_validation_curve(clf, x_train, y_train, 'degree', param_range, param_range,
                              plot_titles_['val_curve_degree'],
                              f'{img_names_["val_curve_degree"]}',
                              'Degree', results_file=results_file_,
                              annotation_text=sc.make_annotation_string(clf, 'degree'))

    # Validation curve for C
    param_range = np.logspace(-3, 1, 20)
    a1c.plot_validation_curve(clf, x_train, y_train, 'C', param_range, param_range,
                              plot_titles_['val_curve_C'],
                              f'{img_names_["val_curve_C"]}',
                              'L2 Regularization', results_file=results_file_,
                              annotation_text=sc.make_annotation_string(clf, 'C'), xscale='log')

    # Step 6. Perform grid search by selecting the range for each parameter to include the values found using
    #         validation curves
    # clf.coef0 = 1
    # grid_params = make_grid_search_params()
    # a1c.perform_grid_search(clf, grid_params, x_train, y_train, verbose=4, write_results=True,
    #                         file_name=gs_results_file_)
    # Observations
    #   Degree 4 and C 0.01 was the best choice, both in terms of performance and Occam's Razor

    clf.kernel = 'poly'
    clf.degree = 4
    clf.coef0 = 1
    clf.C = 0.01
    clf.probability = True

    # Step 8. Plot learning and loss curves of tuned model (using cross-validation on training dataset only)
    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_tuned'],
                            img_names_['learn_curve_tuned'], sc.make_annotation_string(clf))

    # Step 9. Retrain tuned model with ALL training data
    clf.fit(x_train, y_train)

    # Step 10. Finally, confirm with the test dataset
    # Write model performance and Plot PRAUC Curve
    a1c.write_model_performance(clf, x_train, y_train, x_test, y_test, perf_results_file_)
    #       Observations
    #           Average Precision Score = 0.70
    #           Confusion Matrix:
    #               [87  13]
    #               [27 27]

    a1c.plot_prc_curve(clf, x_test, y_test, plot_titles_['pr_curve'], img_names_['pr_curve'])


def make_grid_search_params():
    params = {'kernel': ['poly'],
              'coef0': [1],
              'C': [0.001, 0.01, 0.10, 1],
              'degree': [1, 2, 3, 4, 5]}
    return params


if __name__ == "__main__":
    # np.random.seed(100)
    dataset = 'Wisconsin'
    scoring = a1c.get_scoring()
    img_names_, results_file_, gs_results_file_, perf_results_file_, plot_titles_ = sc.get_utility_maps(dataset)
    run()
