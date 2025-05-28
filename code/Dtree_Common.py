def get_utility_maps(dataset):
    img_names = {}
    plot_titles = {}

    if dataset == 'Pima':
        title_str = 'Pima Diabetes'
    else:
        title_str = 'Wisconsin Breast Cancer'

    img_names['val_curve_max_depth'] = f'DT_validation_curve_max_depth_{dataset}.jpg'
    img_names['val_curve_min_samples_split'] = f'DT_validation_curve_min_samples_split_{dataset}.jpg'
    img_names['val_curve_ccp_alpha_gini'] = f'DT_validation_curve_ccp_alpha_gini_{dataset}.jpg'
    img_names['val_curve_ccp_alpha_entropy'] = f'DT_validation_curve_ccp_alpha_entropy_{dataset}.jpg'
    img_names['val_curve_criterion'] = f'DT_validation_curve_criterion_{dataset}.jpg'
    img_names['val_curve_min_impurity_decrease'] = f'DT_validation_curve_min_impurity_decrease_{dataset}.jpg'
    img_names['val_curve_max_features'] = f'DT_validation_curve_max_features_{dataset}.jpg'

    img_names['learn_curve_untuned'] = f'DT_learning_curve_untuned_{dataset}.jpg'
    img_names['learn_curve_tuned'] = f'DT_learning_curve_tuned_{dataset}.jpg'

    img_names['pr_curve'] = f'DT_Precision_Recall_curve_{dataset}.jpg'

    results_file = f'DT_results_{dataset}.txt'
    gs_results_file = f'DT_gs_results_{dataset}.txt'
    perf_results_file = f'DT_perf_results_{dataset}.txt'

    plot_titles['val_curve_max_depth'] = f'Decision Tree Validation Curve of {title_str} \n(Max Tree Depth ' \
                                         f'v/s ' \
                                         f'Average Precision Score)'
    plot_titles['val_curve_min_samples_split'] = f'Decision Tree Validation Curve of {title_str} \n(Minimum Samples ' \
                                                 f'to Split v/s ' \
                                                 f'Average Precision Score)'
    plot_titles['val_curve_ccp_alpha_gini'] = f'Decision Tree Validation Curve of {title_str} \n(Cost Complexity Path ' \
                                              f'Value (post-pruning with Gini) v/s ' \
                                              f'Average Precision Score)'
    plot_titles[
        'val_curve_ccp_alpha_entropy'] = f'Decision Tree Validation Curve of {title_str} \n(Cost Complexity Path ' \
                                         f'Value (post-pruning with Entropy) v/s ' \
                                         f'Average Precision Score)'
    plot_titles['val_curve_criterion'] = f'Decision Tree Validation Curve of {title_str} \n(Information Gain ' \
                                         f'Criterion P v/s ' \
                                         f'Average Precision Score)'
    plot_titles['val_curve_min_impurity_decrease'] = f'Decision Tree Validation Curve of {title_str} \n(Minimum ' \
                                                     f'Impurity Decrease to Split v/s ' \
                                                     f'Average Precision Score)'
    plot_titles['val_curve_max_features'] = f'Decision Tree Validation Curve of {title_str} \n(Maximum ' \
                                            f'Features to Consider v/s ' \
                                            f'Average Precision Score)'

    plot_titles['learn_curve_untuned'] = f'Decision Tree Learning Curve (Untuned Model) - {title_str}'
    plot_titles['learn_curve_tuned'] = f'Decision Tree Learning Curve (Tuned Model) - {title_str}'

    plot_titles['pr_curve'] = f'Decision Tree Precision Recall Curve of {title_str}'

    return img_names, results_file, gs_results_file, perf_results_file, plot_titles


def make_annotation_string(clf, param_to_exclude=None):
    return_str = ''
    if 'ccp_alpha' != param_to_exclude:
        return_str = ''.join([return_str, f'ccp_alpha={clf.ccp_alpha}'])

    if 'criterion' != param_to_exclude:
        return_str = '\n'.join([return_str, f'criterion: {clf.criterion}'])

    return return_str
