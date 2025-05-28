import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import A1_Common as a1c
import KNN_Common as knc


def run():
    # Step 1. Split into training and testing datasets
    x_train, x_test, y_train, y_test = a1c.get_preprocessed_train_test_data(dataset)

    # Step 2. Generate learning curve for base model
    clf = KNeighborsClassifier()
    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_untuned'],
                            img_names_['learn_curve_untuned'], annotation=knc.make_annotation_string(clf))

    # # Step 3. Validation curve to determine impact of "k" (Hyperparameter n_neighbors)
    param_range = [i for i in range(1, 51, 1)]
    a1c.plot_validation_curve(clf, x_train, y_train, 'n_neighbors', param_range, param_range,
                              plot_titles_['val_curve_k'], img_names_['val_curve_k'], 'K', results_file_,
                              annotation_text=knc.make_annotation_string(clf, 'k'))
    #
    # # Step 4. Validation curve to determine impact of weights
    param_range = ['uniform', 'distance']
    a1c.plot_validation_curve(clf, x_train, y_train, 'weights', param_range, param_range,
                              plot_titles_['val_curve_weights'], img_names_['val_curve_weights'],
                              'Weights', results_file_,
                              annotation_text=knc.make_annotation_string(clf, 'weights'))

    # Step 5. choose based on validation curves
    clf.n_neighbors = 24
    clf.weights = 'uniform'

    # Step 6. Retrain tuned model with ALL training data
    clf.fit(x_train, y_train)

    # Step 7. Plot learning and loss curves of tuned model (using cross-validation on training dataset only)
    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_tuned'],
                            f'{img_names_["learn_curve_tuned"]}', annotation=knc.make_annotation_string(clf))

    # Step 8. Finally, confirm with the test dataset
    # Write model performance and Plot PRAUC Curve
    a1c.write_model_performance(clf, x_train, y_train, x_test, y_test, perf_results_file_)

    a1c.plot_prc_curve(clf, x_test, y_test, plot_titles_['pr_curve'], img_names_['pr_curve'])


def make_grid_search_params():
    return {'n_neighbors': [n for n in range(5, 51, 1)],
            'weights': ['uniform', 'distance']}


if __name__ == "__main__":
    np.random.seed(100)
    dataset = 'Pima'
    scoring = a1c.get_scoring()
    img_names_, results_file_, gs_results_file_, perf_results_file_, plot_titles_ = knc.get_utility_maps(dataset)
    run()
