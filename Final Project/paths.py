from pathlib import Path

# parent directory
repo_dir = Path(__file__).absolute().parent

# directory containing data files
data_dir = repo_dir / 'data/'

# directory containing results
results_dir = repo_dir / 'results/'

# directory containing trained models
model_dir = repo_dir / 'models/'

# create directories that don't exist
for f in [data_dir, results_dir]:
    f.mkdir(exist_ok = True)
