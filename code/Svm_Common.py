def get_utility_maps(dataset):
    img_names = {}
    plot_titles = {}

    if dataset == 'Pima':
        title_str = 'Pima Diabetes'
    else:
        title_str = 'Wisconsin Breast Cancer'

    img_names['val_curve_kernel'] = f'Svm_validation_curve_kernel_{dataset}.jpg'
    img_names['val_curve_C'] = f'Svm_validation_curve_c_{dataset}.jpg'
    img_names['val_curve_gamma'] = f'Svm_validation_curve_gamma_{dataset}.jpg'
    img_names['val_curve_degree'] = f'Svm_validation_curve_degree_{dataset}.jpg'

    img_names['learn_curve_untuned'] = f'Svm_learning_curve_untuned_{dataset}.jpg'
    img_names['learn_curve_tuned'] = f'Svm_learning_curve_tuned_{dataset}.jpg'

    img_names['pr_curve'] = f'Svm_Precision_Recall_curve_{dataset}.jpg'

    results_file = f'Svm_results_{dataset}.txt'
    gs_results_file = f'Svm_gs_results_{dataset}.txt'
    perf_results_file = f'Svm_perf_results_{dataset}.txt'

    plot_titles['val_curve_kernel'] = f'SVM Validation Curve of {title_str} \n(Kernel ' \
                                      f'v/s ' \
                                      f'Average Precision Score)'
    plot_titles['val_curve_C'] = f'SVM Validation Curve of {title_str} \n(L2 Regularization "C" ' \
                                 f'v/s ' \
                                 f'Average Precision Score)'
    plot_titles['val_curve_gamma'] = f'SVM Validation Curve of {title_str} \n(Kernel Coefficient Gamma ' \
                                     f'v/s Average Precision Score)'
    plot_titles['val_curve_degree'] = f'SVM Validation Curve of {title_str} \n(Polynomial Degree ' \
                                      f'v/s Average Precision Score)'

    plot_titles['learn_curve_untuned'] = f'SVM Learning Curve (Untuned Model) - {title_str}'
    plot_titles['learn_curve_tuned'] = f'SVM Learning Curve (Tuned Model) - {title_str}'

    plot_titles['pr_curve'] = f'SVM Precision Recall Curve of {title_str}'

    return img_names, results_file, gs_results_file, perf_results_file, plot_titles


def make_annotation_string(clf, param_to_exclude=None):
    return_str = ''
    if 'kernel' != param_to_exclude:
        return_str = ''.join([return_str, f'kernel={clf.kernel}'])

    if 'C' != param_to_exclude:
        return_str = '\n'.join([return_str, f'C={clf.C}'])

    if 'gamma' != param_to_exclude:
        return_str = '\n'.join([return_str, f'gamma={clf.gamma}'])

    if 'degree' != param_to_exclude:
        return_str = '\n'.join([return_str, f'degree={clf.degree}'])

    return return_str
