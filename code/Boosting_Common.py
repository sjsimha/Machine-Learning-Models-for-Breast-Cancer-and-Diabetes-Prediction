def get_utility_maps(dataset):
    img_names = {}
    plot_titles = {}

    if dataset == 'Pima':
        title_str = 'Pima Diabetes'
    else:
        title_str = 'Wisconsin Breast Cancer'

    img_names['val_curve_n'] = f'Boost_validation_curve_n_{dataset}.jpg'
    img_names['val_curve_learn_rate'] = f'Boost_validation_curve_learn_rate_{dataset}.jpg'
    img_names['val_curve_algorithm'] = f'Boost_validation_curve_algorithm_{dataset}.jpg'

    img_names['learn_curve_untuned'] = f'Boost_learning_curve_untuned_{dataset}.jpg'
    img_names['learn_curve_tuned'] = f'Boost_learning_curve_tuned_{dataset}.jpg'

    img_names['pr_curve'] = f'Boost_Precision_Recall_curve_{dataset}.jpg'

    results_file = f'Boost_results_{dataset}.txt'
    gs_results_file = f'Boost_gs_results_{dataset}.txt'
    perf_results_file = f'Boost_perf_results_{dataset}.txt'

    plot_titles['val_curve_n'] = f'Boosting Validation Curve of {title_str} \n(Number of Estimators ' \
                                 f'v/s ' \
                                 f'Average Precision Score)'
    plot_titles['val_curve_learn_rate'] = f'Boosting Validation Curve of {title_str} \n(Learning Rate ' \
                                          f'v/s ' \
                                          f'Average Precision Score)'
    plot_titles['val_curve_algorithm'] = f'Boosting Validation Curve of {title_str} \n(Boosting Algorithm ' \
                                         f'Average Precision Score)'

    plot_titles['learn_curve_untuned'] = f'Boosting Learning Curve (Untuned Model) - {title_str}'
    plot_titles['learn_curve_tuned'] = f'Boosting Learning Curve (Tuned Model) - {title_str}'

    plot_titles['pr_curve'] = f'Boosting Precision Recall Curve of {title_str}'

    return img_names, results_file, gs_results_file, perf_results_file, plot_titles


def make_annotation_string(clf, param_to_exclude=None):
    return_str = ''
    if 'n_estimators' != param_to_exclude:
        return_str = ''.join([return_str, f'n_estimators={clf.n_estimators}'])

    if 'learning_rate' != param_to_exclude:
        return_str = '\n'.join([return_str, f'learning_rate={clf.learning_rate}'])

    if 'algorithm' != param_to_exclude:
        return_str = '\n'.join([return_str, f'algorithm={clf.algorithm}'])

    return return_str
