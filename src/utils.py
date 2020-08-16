import os

BUILD_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
PLOTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plots'))
TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train'))

AUC_PATH = os.path.join(PLOTS_PATH, 'auc_vs_depth')
LFD_PATH = os.path.join(PLOTS_PATH, 'lfd_vs_depth')


def make_folders(dataset, metric, method):
    dir_paths = [f'../data',
                 f'../data/{dataset}',
                 f'../data/{dataset}/plots-98',
                 f'../data/{dataset}/plots-98/{metric}',
                 f'../data/{dataset}/plots-98/{metric}/{method}']
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
    return


def manifold_path(dataset, metric, min_points, graph_ratio) -> str:
    """ Generate proper path to manifold. """
    return os.path.join(
        BUILD_PATH,
        ':'.join(map(str, [dataset, metric, min_points, f'{graph_ratio}.pickle']))
    )


def dataset_from_path(path):
    return os.path.basename(path).split(':')[0]