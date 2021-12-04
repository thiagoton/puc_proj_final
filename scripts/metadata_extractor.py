import csv
import os

ALLOWED_CLASSES = {
    'horn': 0,
    'siren': 1,
    'noise': 3
}


def translate_label(translation_map, input_label):
    return translation_map[input_label]


class UrbanSound8kExtractor:
    def __init__(self, metadata_path) -> None:
        self.metadata_path = metadata_path
        self.samples_names = []
        self.samples_labels = []

    def class_translation_map(self):
        return {
            'air_conditioner': 'noise',
            'car_horn': 'horn',
            'children_playing': 'noise',
            'dog_bark': 'noise',
            'drilling': 'noise',
            'engine_idling': 'noise',
            'gun_shot': 'noise',
            'jackhammer': 'noise',
            'siren': 'siren',
            'street_music': 'noise',
        }

    def load_metadata(self):
        with open(self.metadata_path) as f:
            reader = csv.reader(f, delimiter=',')
            for line in reader:
                self.samples_names.append(line[0])
                self.samples_labels.append(line[-1])

    def get_class(self, input_sample):
        '''
        Returns the class associated with given input sample
        '''

        input_sample_name = os.path.basename(input_sample) 
        try:
            index = self.samples_names.index(input_sample_name)
            return self.samples_labels[index]
        except:
            return None

