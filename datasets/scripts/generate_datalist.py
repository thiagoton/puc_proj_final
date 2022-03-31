import argparse
import os
import sys
import random

from common import media_descriptor  # nopep8
from common import media_audio  # nopep8
from common import utils  # nopep8
from common.metadata_extractor import *  # nopep8


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generates datalist to be used during training/test')
    parser.add_argument('--train_split', type=int, default=70,
                        help='Amount (percentage) on files to be used for training (default=%(default))')
    parser.add_argument('output_dir', type=str,
                        help='Place to save output lists')
    parser.add_argument('input_dirs', type=str, nargs='+',
                        help='List of input directories to search for input files to be used for training/test')
    return parser.parse_args()


def generate_train_list(splitted_samples, split_precentage):
    train_sample_list = []
    test_sample_list = []
    max_samples_per_class = None
    for label in splitted_samples.keys():
        num_samples = len(splitted_samples[label])
        if max_samples_per_class is None:
            max_samples_per_class = num_samples

        max_samples_per_class = min(max_samples_per_class, num_samples)

    train_samples_per_class = int(split_precentage/100. * max_samples_per_class)
    test_samples_per_class = int((100 - split_precentage)/100. * max_samples_per_class)
    for label in splitted_samples.keys():
        samples = splitted_samples[label]
        random.shuffle(samples)
        train_sample_list.extend(samples[:train_samples_per_class])
        test_sample_list.extend(samples[train_samples_per_class:(train_samples_per_class+test_samples_per_class)])

    return train_sample_list, test_sample_list

def split_samples(list_files):
    samples_for_label = {}

    for f in list_files:
        label = f['label']
        if label not in samples_for_label.keys():
            samples_for_label[label] = []
        samples_for_label[label].append(f)

    return samples_for_label


def search_for_files(input_dir):
    file_list = []
    list_of_files = media_audio.list_media_files(input_dir)

    for file in list_of_files:
        entry = {}
        entry['inputfile'] = file
        metadata_path = utils.make_path_absolute(file.replace('.wav', '.json'))
        metadata = media_descriptor.MediaDescriptor(metadata_path).read()
        entry.update(metadata)
        file_list.append(entry)

    return file_list


def save_list(items, output_dir, fname):
    fpath = os.path.join(output_dir, fname)
    with open(fpath, 'w') as f:
        for e in items:
            f.write(e['inputfile'] + os.linesep)

if __name__ == '__main__':
    args = parse_args()

    train_split = args.train_split
    output_dir = args.output_dir
    input_dirs = args.input_dirs

    files = []
    for dir in input_dirs:
        files.extend(search_for_files(dir))

    splitted_samples = split_samples(files)
    train_list, test_list = generate_train_list(splitted_samples, train_split)

    save_list(train_list, output_dir, 'trainlist.txt')
    save_list(test_list, output_dir, 'testlist.txt')
