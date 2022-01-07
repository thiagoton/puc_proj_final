import os
import yaml

PROJECT_PATH = os.path.realpath(os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..'))

def make_path_relative(input_path):
    return os.path.relpath(os.path.realpath(input_path), PROJECT_PATH)

def make_path_absolute(input_path):
    return os.path.join(PROJECT_PATH, input_path)

def load_params(params_file='params.yaml'):
    with open(params_file) as fd:
        return yaml.safe_load(fd)