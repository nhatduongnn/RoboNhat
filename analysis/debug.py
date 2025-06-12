import sys
import os
sys.path.insert(0, '../pkg/')
sys.path.insert(0, '/home/rapiduser/programs/RoboCOP/pkg/')
#os.environ["R_HOME"] = '/home/rapiduser/miniconda3/envs/robocop-2024/'
from run_robocop import run_robocop_with_em, run_robocop_without_em, plot_robocop_output


configfile = './config_example.ini'
coord_file_train = './coord_train.tsv'
coord_file_all = './coord_all.tsv'
# Output directories for the RoboCOP runs
outdir_train = './robocop_train/'
outdir_all = './robocop_all/'
outdir_all_subset = './robocop_all_subset/'


run_robocop_with_em(coord_file_train, configfile, outdir_train)
