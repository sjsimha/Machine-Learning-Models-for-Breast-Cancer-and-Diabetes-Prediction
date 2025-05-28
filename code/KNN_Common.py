import A1_Common as a1c


def get_utility_maps(dataset):
    img_names = {}
    plot_titles = {}

    if dataset == 'Pima':
        title_str = 'Pima Diabetes'
    else:
        title_str = 'Wisconsin Breast Cancer'

    img_names['val_curve_k'] = f'KNN_validation_curve_k_{dataset}.jpg'
    img_names['val_curve_leaf_size'] = f'KNN_validation_curve_leaf_size_{dataset}.jpg'
    img_names['val_curve_p'] = f'KNN_validation_curve_p_{dataset}.jpg'
    img_names['val_curve_weights'] = f'KNN_validation_curve_weights_{dataset}.jpg'
    img_names['val_curve_algorithm'] = f'KNN_validation_curve_algorithm_{dataset}.jpg'

    img_names['learn_curve_untuned'] = f'KNN_learning_curve_untuned_{dataset}.jpg'
    img_names['learn_curve_tuned'] = f'KNN_learning_curve_tuned_{dataset}.jpg'

    img_names['pr_curve'] = f'KNN_Precision_Recall_curve_{dataset}.jpg'

    results_file = f'KNN_results_{dataset}.txt'
    gs_results_file = f'KNN_gs_results_{dataset}.txt'
    perf_results_file = f'KNN_perf_results_{dataset}.txt'

    plot_titles['val_curve_k'] = f'KNN Validation Curve of {title_str} \n(# of Nearest Neighbors v/s ' \
                                 f'Average Precision Score)'
    plot_titles['val_curve_leaf_size'] = f'KNN Validation Curve of {title_str} \n(Leaf Size v/s ' \
                                         f'Average Precision Score)'
    plot_titles['val_curve_p'] = f'KNN Validation Curve of {title_str} \n(Power Parameter P v/s ' \
                                 f'Average Precision Score)'
    plot_titles['val_curve_weights'] = f'KNN Validation Curve of {title_str} \n(Weights v/s ' \
                                       f'Average Precision Score)'
    plot_titles['val_curve_algorithm'] = f'KNN Validation Curve of {title_str} \n(Algorithm v/s ' \
                                         f'Average Precision Score)'

    plot_titles['learn_curve_untuned'] = f'K Nearest Neighbors Learning Curve (Untuned Model) - {title_str}'
    plot_titles['learn_curve_tuned'] = f'K Nearest Neighbors Learning Curve (Tuned Model) - {title_str}'

    plot_titles['pr_curve'] = f'K Nearest Neighbors Precision Recall Curve of {title_str}'

    return img_names, results_file, gs_results_file, perf_results_file, plot_titles


def make_annotation_string(clf, param_to_exclude=None):
    return_str = ''
    if 'k' != param_to_exclude:
        return_str = ''.join([return_str, f'k={clf.n_neighbors}'])

    if 'weights' != param_to_exclude:
        return_str = '\n'.join([return_str, f'weights={clf.weights}'])

    return return_str


def plot_validation_curve_k(clf, x_train, y_train, title, img_name, results_file, iter_range):
    annotation = make_annotation_string(clf)
    a1c.plot_validation_curve(clf, x_train, y_train, 'n_neighbors', iter_range, iter_range,
                              title, img_name, 'K ---->', results_file, annotation_text=annotation)


def plot_validation_curve_leaf_size(clf, x_train, y_train, title, img_name, results_file, iter_range):
    annotation = make_annotation_string(clf)
    a1c.plot_validation_curve(clf, x_train, y_train, 'leaf_size', iter_range, iter_range,
                              title, img_name, 'Leaf Size ---->', results_file, annotation_text=annotation)


def plot_validation_curve_p(clf, x_train, y_train, title, img_name, results_file, iter_range):
    annotation = make_annotation_string(clf)
    a1c.plot_validation_curve(clf, x_train, y_train, 'p', iter_range, iter_range,
                              title, img_name, 'Power Parameter p ---->', results_file,
                              annotation_text=annotation)


def plot_validation_curve_weights(clf, x_train, y_train, title, img_name, results_file, iter_range):
    a1c.plot_validation_curve(clf, x_train, y_train, 'weights', iter_range, iter_range, title, img_name,
                              'Weights ---->', results_file, annotation_text=make_annotation_string(clf))


def plot_validation_curve_algorithm(clf, x_train, y_train, title, img_name, results_file, iter_range):
    a1c.plot_validation_curve(clf, x_train, y_train, 'algorithm', iter_range, iter_range, title, img_name,
                              'Algorithm ---->', results_file, annotation_text=make_annotation_string(clf))
