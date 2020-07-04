import os


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
        LOG_DIR,
        ':'.join(map(str, [dataset, metric, min_points, f'{graph_ratio}.pickle']))
    )


def dataset_from_path(path):
    return os.path.basename(path).split(':')[0]