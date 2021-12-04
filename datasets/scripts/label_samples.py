import os
import sys

# import infrastructure
ROOT_SCRIPTS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'scripts'))
sys.path.append(ROOT_SCRIPTS_PATH)
import media_descriptor  # nopep8
from metadata_extractor import *  # nopep8


def load_samples(folder):
    extractor = UrbanSound8kExtractor(
        '../samples/raw/UrbanSound8K/metadata/UrbanSound8K.csv')
    extractor.load_metadata()
    for root, dirs, files in os.walk(folder):
        for file in files:
            if not '.json' in file:
                continue
            path = os.path.join(root, file)
            descriptor = media_descriptor.MediaDescriptor(path)
            data = descriptor.read()

            label = extractor.get_class(data['raw_filepath'])
            data['label'] = translate_label(extractor.class_translation_map(), label)
            descriptor.update(data)


if __name__ == '__main__':
    load_samples('../samples/preprocessed')
