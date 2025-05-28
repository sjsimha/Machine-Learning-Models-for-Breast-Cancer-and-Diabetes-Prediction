import numpy as np
from sklearn.neural_network import MLPClassifier

import A1_Common as a1c
import NN_Common as nnc


def run():
    # Step 1. Split into training and testing datasets, and create base classifier
    x_train, x_test, y_train, y_test = a1c.get_preprocessed_train_test_data(dataset)
    clf = MLPClassifier(random_state=12)

    # Step 2. Generate learning and loss curves for base model
    nnc.plot_learning_and_loss_curves(clf, x_train, y_train, plot_titles_['learn_curve_untuned'],
                                      img_names_['learn_curve_untuned'], plot_titles_['loss_curve_untuned'],
                                      img_names_['loss_curve_untuned'])

    # Step 5. Validation curve to determine impact of  nodes per layer (Hyperparamer hidden_layer_sizes)
    nnc.plot_validation_curve_neurons(clf, x_train, y_train, 1, plot_titles_['val_curve_neurons'],
                                      img_names_['val_curve_neurons'], results_file_)
    # Observations
    #     * Choose neurons = 12
    clf.hidden_layer_sizes = (12,)

    batch_range = [8, 16, 32, 64, 128, 256]
    a1c.plot_validation_curve(clf, x_train, y_train, 'batch_size', batch_range, batch_range,
                              plot_titles_['val_curve_batch'],
                              img_names_['val_curve_batch'], 'Batch Size', results_file_,
                              annotation_text=nnc.make_annotation_string(clf))
    clf.batch_size = 64

    # Step 6. Validation curve to determine impact of L2 Regularization (Hyperparamer alpha) alpha_range
    alpha_range = np.logspace(-3, 1, 10)
    a1c.plot_validation_curve(clf, x_train, y_train, 'alpha', alpha_range, alpha_range,
                              plot_titles_['val_curve_alpha'], img_names_['val_curve_alpha'], 'alpha', results_file_,
                              annotation_text=nnc.make_annotation_string(clf), xscale='log')
    clf.alpha = 0.4

    # Step 7. Plot learning and loss curves of tuned model (using cross-validation on training dataset only)
    nnc.plot_learning_and_loss_curves(clf, x_train, y_train, plot_titles_['learn_curve_tuned'],
                                      img_names_['learn_curve_tuned'], plot_titles_['loss_curve_tuned'],
                                      img_names_['loss_curve_tuned'])

    # Step 8. Perform grid search by selecting the range for each parameter to include the values found using
    #         validation curves
    # a1c.perform_grid_search(clf, make_grid_search_params(), x_train, y_train, verbose=4, write_results=True,
    #                         file_name=gs_results_file_)

    clf.hidden_layer_sizes = (12,)
    clf.batch_size = 64
    clf.alpha = 0.4

    # Step 9. Retrain tuned model with ALL training data
    clf.fit(x_train, y_train)
    nnc.plot_learning_and_loss_curves(clf, x_train, y_train, plot_titles_['learn_curve_tuned'],
                                      img_names_['learn_curve_tuned'], plot_titles_['loss_curve_tuned'],
                                      img_names_['loss_curve_tuned'])

    # Step 10. Finally, confirm with the test dataset
    # Write model performance and Plot PRAUC Curve
    a1c.write_model_performance(clf, x_train, y_train, x_test, y_test, perf_results_file_)
    a1c.plot_prc_curve(clf, x_test, y_test, plot_titles_['pr_curve'], img_names_['pr_curve'])


def make_grid_search_params():
    params = {'max_iter': [itr for itr in range(1000, 5000, 1000)]}

    hidden_layer_sizes = []
    for i in range(1, 5 + 1):
        for j in range(2, 10 + 1):
            hidden_layer_sizes.append((tuple(np.ones(i, int) * j)))
    params['hidden_layer_sizes'] = hidden_layer_sizes

    params['batch_size'] = [8, 16, 32, 64, 128, 256]
    params['alpha'] = [alpha for alpha in range(3, 10 + 1)]

    return params


if __name__ == "__main__":
    np.random.seed(100)
    dataset = 'Pima'
    scoring = a1c.get_scoring()
    img_names_, results_file_, gs_results_file_, perf_results_file_, plot_titles_ = nnc.get_utility_maps('Pima')
    run()
