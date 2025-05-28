import time
import numpy as np
from sklearn import ensemble, tree
import A1_Common as a1c
import Boosting_Common as bc


def run():
    # Step 1. Split into training and testing datasets
    x_train, x_test, y_train, y_test = a1c.get_preprocessed_train_test_data(dataset)

    # Step 2. Generate learning curve for base model;
    clf = ensemble.AdaBoostClassifier(random_state=1,
                                      estimator=tree.DecisionTreeClassifier(random_state=1, max_depth=1))

    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_untuned'],
                            img_names_['learn_curve_untuned'], bc.make_annotation_string(clf))
    #   Observations
    #       Base model exhibits no bias and low Variance
    #           Training Average Precision score = 1.0
    #           Validation Set Average Precision score ~0.97
    #       More data likely not needed
    #

    # Step 3. Validation curve to determine impact of number of estimators (Hyperparameter n_estimators)
    param_range = [i for i in range(1, 101, 1)]
    a1c.plot_validation_curve(clf, x_train, y_train, 'n_estimators', param_range, param_range,
                              plot_titles_['val_curve_n'], img_names_['val_curve_n'],
                              'Number of Base Estimators',
                              results_file=results_file_,
                              annotation_text=bc.make_annotation_string(clf, 'n_estimators'))

    #       Observations
    #           * Validation score stabilizes around 31 with minute increases afterwards and reaching maximum value at
    #           * n_estimators = 79
    #           * Gridsearch recommendation: 20 to 100

    # Step 4. Validation curve to determine impact of "learning rate" (Hyperparameter learning_rate)
    param_range = np.linspace(0.1, 1, 10)
    a1c.plot_validation_curve(clf, x_train, y_train, 'learning_rate', param_range, param_range,
                              plot_titles_['val_curve_learn_rate'], img_names_['val_curve_learn_rate'],
                              'Learning Rate',
                              results_file=results_file_,
                              annotation_text=bc.make_annotation_string(clf, 'learning_rate'))

    #       Observations
    #           * Validation set score is mostly at the peak from 0.1 to 0.4, after which overfitting sets in
    #           * Gridsearch recommendation: 0.1 to 1.0

    # Step 3. Validation curve to determine interplay between learning rate and estimators
    for ln in [0.1, 0.4]:
        clf.learning_rate = ln
        param_range = [i for i in range(1, 20, 1)]
        a1c.plot_validation_curve(clf, x_train, y_train, 'n_estimators', param_range, param_range,
                                  plot_titles_['val_curve_n'], f'{img_names_["val_curve_n"]}_{ln}',
                                  'Number of Base Estimators',
                                  results_file=results_file_,
                                  annotation_text=bc.make_annotation_string(clf, 'n_estimators'))

    # Step 6. Perform grid search by selecting the range for each parameter to include the values found using
    #         validation curves
    # a1c.perform_grid_search(clf, make_grid_search_params(), x_train, y_train, verbose=4, write_results=True,
    #                         file_name=gs_results_file_)
    # Observations
    #   * Top 10 configurations had close performance scores only differing in the 3rd or 4th decimal position,
    #   * all with the same learning rate of 0.2, the default SAMME.R algorithm, and the least number of estimators
    #   * was 73, chosen based on Occam's razor
    #   * The following hyperparameter values are chosen:

    #           n_estimators = 73
    #           learning_rate = 0.20
    clf.n_estimators = 60
    clf.learning_rate = 0.20

    # Step 8. Plot learning and loss curves of tuned model (using cross-validation on training dataset only)
    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_tuned'],
                            img_names_['learn_curve_tuned'], bc.make_annotation_string(clf))

    # Step 9. Retrain tuned model with ALL training data
    clf.fit(x_train, y_train)

    # Step 10. Finally, confirm with the test dataset
    # Write model performance and Plot PRAUC Curve
    a1c.write_model_performance(clf, x_train, y_train, x_test, y_test, perf_results_file_)
    #       Observations
    #           Average Precision Score = 0.9958
    #           Confusion Matrix:
    #               [88  1]
    #               [2 46]

    a1c.plot_prc_curve(clf, x_test, y_test, plot_titles_['pr_curve'], img_names_['pr_curve'])


def make_grid_search_params():
    params = {'n_estimators': [n for n in range(20, 101, 1)],
              'learning_rate': np.linspace(0.10, 1.0, 10)}

    return params


if __name__ == "__main__":
    np.random.seed(100)
    dataset = 'Wisconsin'
    scoring = a1c.get_scoring()
    img_names_, results_file_, gs_results_file_, perf_results_file_, plot_titles_ = bc.get_utility_maps(dataset)
    run()
