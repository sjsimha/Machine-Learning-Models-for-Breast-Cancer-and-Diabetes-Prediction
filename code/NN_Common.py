import pandas as pd
from matplotlib import pyplot as plt
import A1_Common as a1c
import numpy as np
import datetime as dt
from sklearn.model_selection import learning_curve, validation_curve


def get_utility_maps(dataset):
    img_names = {}
    plot_titles = {}

    if dataset == 'Pima':
        title_str = 'Pima Diabetes'
    else:
        title_str = 'Wisconsin Breast Cancer'

    img_names['val_curve_hlayers'] = f'NN_validation_curve_hlayers_{dataset}.jpg'
    img_names['val_curve_neurons'] = f'NN_validation_curve_neurons_{dataset}.jpg'
    img_names['val_curve_activations'] = f'NN_validation_curve_activations_{dataset}.jpg'
    img_names['val_curve_iters'] = f'NN_validation_curve_maxiters_{dataset}.jpg'
    img_names['val_curve_alpha'] = f'NN_validation_curve_alpha_{dataset}.jpg'
    img_names['val_curve_batch'] = f'NN_validation_curve_batch_{dataset}.jpg'
    img_names['val_curve_learning_rate_init'] = f'NN_validation_curve_learning_rate_init_{dataset}.jpg'

    img_names['learn_curve_untuned'] = f'NN_learning_curve_untuned_{dataset}.jpg'
    img_names['loss_curve_untuned'] = f'NN_loss_curve_untuned_{dataset}.jpg'

    img_names['learn_curve_tuned'] = f'NN_learning_curve_tuned_{dataset}.jpg'
    img_names['loss_curve_tuned'] = f'NN_loss_curve_tuned_{dataset}.jpg'

    img_names['pr_curve'] = f'NN_Precision_Recall_curve_{dataset}.jpg'

    results_file = f'NN_results_{dataset}.txt'
    gs_results_file = f'NN_gs_results_{dataset}.txt'
    perf_results_file = f'NN_perf_results_{dataset}.txt'

    plot_titles['val_curve_hlayers'] = f'Neural Network Validation Curve of {title_str} \n(Hidden Layer Size v/s ' \
                                       f'Average Precision Score)'
    plot_titles['val_curve_neurons'] = f'Neural Network Validation Curve of {title_str}  \n(Neurons per Layer ' \
                                       f'v/s Average Precision Score)'
    plot_titles['val_curve_activations'] = f'Neural Network Validation Curve of {title_str}  \n(Comparison of  ' \
                                           f'Activation Functions v/s Average Precision Score)'
    plot_titles['val_curve_iters'] = f'Neural Network Validation Curve of {title_str}  \n(Comparison of  ' \
                                     f'Training Epochs v/s Average Precision Score)'
    plot_titles['val_curve_alpha'] = f'Neural Network Validation Curve of {title_str}  \n(Comparison of  ' \
                                     f'L2 Regularization Factor v/s Average Precision Score)'
    plot_titles['val_curve_batch'] = f'Neural Network Validation Curve of {title_str}  \n(Comparison of  ' \
                                     f'Batch Size v/s Average Precision Score)'
    plot_titles['val_curve_learning_rate_init'] = f'Neural Network Validation Curve of {title_str}  \n(Comparison of  ' \
                                                  f'Learning Rate v/s Average Precision Score)'

    plot_titles['learn_curve_untuned'] = f'Neural Network Learning Curve (Untuned Model) - {title_str}'
    plot_titles['learn_curve_tuned'] = f'Neural Network Learning Curve (Tuned Model) - {title_str}'

    plot_titles['loss_curve_tuned'] = f'Neural Network Loss Curve (Tuned Model) - {title_str}'
    plot_titles['loss_curve_untuned'] = f'Neural Network Loss Curve (Untuned Model) - {title_str}'

    plot_titles['pr_curve'] = f'Neural Network Precision Recall Curve of {title_str}'

    return img_names, results_file, gs_results_file, perf_results_file, plot_titles


def make_annotation_string(clf, param_to_exclude=None):
    return_str = ''
    if 'hidden_layer_sizes' != param_to_exclude:
        return_str = ''.join([return_str, f'hidden_layer_sizes={clf.hidden_layer_sizes}'])

    # if 'alpha' != param_to_exclude:
    #     return_str = '\n'.join([return_str, f'alpha={clf.alpha}'])

    if 'batch_size' != param_to_exclude:
        return_str = '\n'.join([return_str, f'batch_size={clf.batch_size}'])

    if 'learning_rate_init' != param_to_exclude:
        return_str = '\n'.join([return_str, f'learning_rate_init={clf.learning_rate_init}'])

    return return_str


def plot_learning_and_loss_curves(clf, x_train, y_train, learning_curve_title, learning_curve_img_name,
                                  loss_curve_title, loss_curve_img_name):
    # First, plot learning curve
    plot_learning_curve(clf, x_train, y_train, learning_curve_title, learning_curve_img_name,
                        make_annotation_string(clf))

    # Next, fit model and plot loss curve
    clf.fit(x_train, y_train)
    plot_loss_curve(clf, loss_curve_title, loss_curve_img_name, False)


def plot_learning_curve(clf, x_train, y_train, title, img_name, annotation_text=''):
    train_sizes, train_scores, test_scores = learning_curve(clf, x_train, y_train, scoring=a1c.get_scoring(), cv=5,
                                                            train_sizes=np.linspace(0.1, 1.0, 10), shuffle=True)

    a1c.plot_two_series(train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1),
                        title, '# of Samples', 'Average Precision', 'Training Score',
                        'Testing Score', img_name, save_to_file=True, show_plot=False, annotation_text=annotation_text)


def plot_loss_curve(clf, title, img_name, show_plot=False):
    lc = clf.loss_curve_
    fig, ax = plt.subplots()
    ax.plot(lc)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    ax.text(0.5, 0.5, make_annotation_string(clf), transform=ax.transAxes,
            fontsize=14, color='gray', alpha=0.5,
            ha='center', va='center', rotation=0)
    plt.savefig(f'results/{img_name}')
    if show_plot:
        plt.show()


def plot_validation_curve(clf, x_train, y_train, param_name, param_range, x_axis_range, title, file_name, x_label,
                          results_file, annotation_text=''):
    train_scores, test_scores = validation_curve(clf, x_train, y_train, param_name=param_name, cv=5,
                                                 scoring=a1c.get_scoring(),
                                                 param_range=param_range)

    x_ = np.asarray(x_axis_range).reshape(len(x_axis_range), 1)
    y1_ = np.mean(train_scores, axis=1)
    y1_ = y1_.reshape(len(y1_), 1)
    y2_ = np.mean(test_scores, axis=1)
    y2_ = y2_.reshape(len(y2_), 1)

    with open(f'results/{results_file}', 'a') as sf:
        sf.write(f'\n****Validation Curve Results {title} {annotation_text} '
                 f'\n{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ****\n')
        results_ = np.concatenate((x_, y1_, y2_), axis=1)
        df = pd.DataFrame(results_)
        df.to_csv(sf)

    a1c.plot_two_series(x_axis_range, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1),
                        title, x_label, 'Average Precision', 'Training Score',
                        'Testing Score', file_name, save_to_file=True, show_plot=False, annotation_text=annotation_text)


def plot_validation_curve_max_iters(clf, x_train, y_train, title, img_name, results_file, iter_range):
    annotation = make_annotation_string(clf)
    plot_validation_curve(clf, x_train, y_train, 'max_iter', iter_range, iter_range,
                          title, img_name, '# of Epochs ---->', results_file, annotation_text=annotation)


def plot_validation_curve_hidden_layers(clf, x_train, y_train, title, img_name, results_file, neurons_=None):
    if neurons_ is None:
        neurons = clf.hidden_layer_sizes[0]
    else:
        neurons = neurons_

    # Try up to 5 hidden layers
    layers_range = [(neurons,),
                    (neurons, neurons,),
                    (neurons, neurons, neurons,),
                    (neurons, neurons, neurons, neurons,),
                    (neurons, neurons, neurons, neurons, neurons)]

    x_axis_range = [i for i in range(1, 6)]

    annotation = make_annotation_string(clf)
    plot_validation_curve(clf, x_train, y_train, 'hidden_layer_sizes', layers_range, x_axis_range,
                          title, img_name, '# of Hidden Layers ---->', results_file, annotation_text=annotation)


def plot_validation_curve_neurons(clf, x_train, y_train, layers, title, img_name, results_file, max_neurons=20):
    neurons_range = make_neurons_range(layers, max_neurons)
    plot_validation_curve(clf, x_train, y_train, 'hidden_layer_sizes', neurons_range,
                          [i for i in range(1, max_neurons + 1)],
                          title, img_name, '# of Neurons ---->', results_file,
                          annotation_text=make_annotation_string(clf, 'hidden_layer_sizes'))


def make_neurons_range(layer_size, max_neurons):
    return_list = []
    for neuron in range(1, max_neurons + 1):
        return_list.append(tuple(np.ones(layer_size, int) * neuron))

    return return_list


def plot_validation_curve_L2Regularization(clf, x_train, y_train, title, img_name, results_file, alpha_range):
    plot_validation_curve(clf, x_train, y_train, 'alpha', alpha_range, alpha_range, title, img_name,
                          'L2 Regularization ---->', results_file, annotation_text=make_annotation_string(clf))
