from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT_DIR.joinpath('data')
RESULTS_DIR = ROOT_DIR.joinpath('results')
PLOTS_DIR = ROOT_DIR.joinpath('plots')
UMAPS_DIR = ROOT_DIR.joinpath('umaps')

SCORES_PATH = RESULTS_DIR.joinpath('scores.csv')
TIMES_PATH = RESULTS_DIR.joinpath('times.csv')


def create_required_folders():
    for _dir in (DATA_DIR, RESULTS_DIR, PLOTS_DIR, UMAPS_DIR):
        _dir.mkdir(exist_ok=True)
    return
