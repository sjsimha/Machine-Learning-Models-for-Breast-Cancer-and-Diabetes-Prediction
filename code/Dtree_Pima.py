import numpy as np
from sklearn import tree

import A1_Common as a1c
import Dtree_Common as dtc


def run():
    # Step 1. Split into training and testing datasets
    x_train, x_test, y_train, y_test = a1c.get_preprocessed_train_test_data(dataset)

    # Step 2. Generate learning curve for base model
    clf = tree.DecisionTreeClassifier(random_state=1)
    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_untuned'],
                            img_names_['learn_curve_untuned'], dtc.make_annotation_string(clf))
    #   Observations
    #       Base model exhibits no bias and High Variance
    #           Training Average Precision score = 1
    #           Validation Set Average Precision score ~0.5
    #       More learning data will help since the cross validation learning curve is still going up
    #

    # Step 3. Validation curve to determine impact of post-pruning (Hyperparameter ccp_alpha)
    # This has to be done in two steps separately for gini and entropy

    # Gini
    clf1 = tree.DecisionTreeClassifier(random_state=1, criterion='gini')
    path = clf1.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    a1c.plot_validation_curve(clf1, x_train, y_train, 'ccp_alpha', ccp_alphas, ccp_alphas,
                              plot_titles_['val_curve_ccp_alpha_gini'],
                              img_names_['val_curve_ccp_alpha_gini'],
                              'Cost Complexity Path Alpha',
                              results_file=results_file_, annotation_text=dtc.make_annotation_string(clf1, 'ccp_alpha'),
                              drawstyle='steps-post')

    # Entropy
    clf2 = tree.DecisionTreeClassifier(random_state=1, criterion='entropy')
    path = clf2.cost_complexity_pruning_path(x_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    a1c.plot_validation_curve(clf2, x_train, y_train, 'ccp_alpha', ccp_alphas, ccp_alphas,
                              plot_titles_['val_curve_ccp_alpha_entropy'],
                              img_names_['val_curve_ccp_alpha_entropy'],
                              'Cost Complexity Path Alpha',
                              results_file=results_file_, annotation_text=dtc.make_annotation_string(clf2, 'ccp_alpha'),
                              drawstyle='steps-post')

    # Observations
    #   Gini produced better average precision score, and is faster with training, therefore choose Gini and its
    #   associated best ccp_alpha
    clf.criterion = 'gini'
    clf.ccp_alpha = 0.007043

    # Step 8. Plot learning and loss curves of tuned model (using cross-validation on training dataset only)
    a1c.plot_learning_curve(clf, x_train, y_train, plot_titles_['learn_curve_tuned'],
                            img_names_['learn_curve_tuned'], dtc.make_annotation_string(clf))

    # Step 9. Retrain tuned model with ALL training data
    clf.fit(x_train, y_train)

    # Step 10. Finally, confirm with the test dataset
    # Write model performance and Plot PRAUC Curve
    a1c.write_model_performance(clf, x_train, y_train, x_test, y_test, perf_results_file_)

    a1c.plot_prc_curve(clf, x_test, y_test, plot_titles_['pr_curve'], img_names_['pr_curve'])


if __name__ == "__main__":
    np.random.seed(100)
    dataset = 'Pima'
    scoring = a1c.get_scoring()
    img_names_, results_file_, gs_results_file_, perf_results_file_, plot_titles_ = dtc.get_utility_maps(dataset)

    with open(f'results/{results_file_}', 'w') as sf:
        pass
    run()
