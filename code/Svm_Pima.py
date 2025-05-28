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
    #       Base model exhibits high bias and moderate to high Variance
    #       More learning data might help since the cross validation learning curve is still slightly going up
    #

    # Step 3. Validation curve to determine impact of kernel (Hyperparameter kernel)
    param_range = ['linear', 'poly', 'rbf', 'sigmoid']
    a1c.plot_validation_curve(clf, x_train, y_train, 'kernel', param_range, param_range,
                              plot_titles_['val_curve_kernel'], img_names_['val_curve_kernel'],
                              'Kernel Type',
                              results_file=results_file_,
                              annotation_text=sc.make_annotation_string(clf, 'kernel'))
    #        Observations
    #           * Kernel type showed varying trends.
    #               rbf showed the best bias and the best variance, although both numbers are not optimal
    #           * Although grid search would be recommended to try other kernel and hyperparameter combinations,
    #               rbf is chosen for this assignment
    clf.kernel = 'rbf'

    # Step 5.1 Validation curves to determine impact of Kernel coefficient (Hyperparameter gamma)
    param_range = np.logspace(-5, 1, 20)
    for c in [0.01, 0.1, 1.0, 10]:
        clf.C = c
        a1c.plot_validation_curve(clf, x_train, y_train, 'gamma', param_range, param_range,
                                  plot_titles_['val_curve_gamma'],
                                  f'{img_names_["val_curve_gamma"]}_c{c}',
                                  'Gamma', results_file=results_file_,
                                  annotation_text=sc.make_annotation_string(clf, 'gamma'), xscale='log')
    #  * Gridsearch recommendation: Exhaustive grid search needed

    # Step 6. Perform grid search by selecting the range for each parameter to include the values found using
    #         validation curves
    # a1c.perform_grid_search(clf, make_grid_search_params(), x_train, y_train, verbose=4, write_results=True,
    #                         file_name=gs_results_file_)

    # Observations
    #   * Top 10 configurations had close performance scores around 0.73
    #   *   * Top grid search result was also the least complicated (Ocam's razor)in terms of overfit
    #           (low value of Gamma)
    #   * The following hyperparameter values are chosen:
    #           kernel = 'rbf'
    #           gamma = 0.0297
    #           C = 1.0
    clf.kernel = 'rbf'
    clf.gamma = 0.02976
    clf.C = 1
    clf.probability = True

    # Step 8. Plot learning curves of tuned model (using cross-validation on training dataset only)
    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_tuned'],
                            img_names_['learn_curve_tuned'], sc.make_annotation_string(clf))

    # Step 9. Retrain tuned model with ALL training data
    clf.fit(x_train, y_train)

    # Step 10. Finally, confirm with the test dataset
    # Write model performance and Plot PRAUC Curve
    a1c.write_model_performance(clf, x_train, y_train, x_test, y_test, perf_results_file_)

    a1c.plot_prc_curve(clf, x_test, y_test, plot_titles_['pr_curve'], img_names_['pr_curve'])


def make_grid_search_params():
    params = {'kernel': ['rbf'],
              'C': np.logspace(-1, 0, 10),
              'gamma': np.logspace(-5, 1, 20)
              }
    return params


if __name__ == "__main__":
    dataset = 'Pima'
    scoring = a1c.get_scoring()
    img_names_, results_file_, gs_results_file_, perf_results_file_, plot_titles_ = sc.get_utility_maps(dataset)
    run()
