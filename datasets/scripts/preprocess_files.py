import librosa
import soundfile
import os
import sys
import sys
import argparse

# import infrastructure
ROOT_SCRIPTS_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.path.pardir, os.path.pardir, 'scripts'))
sys.path.append(ROOT_SCRIPTS_PATH)
import media_descriptor  # nopep8
import media_audio  # nopep8
import utils # nopep8


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Preprocess audio files before they are used during training")
    parser.add_argument(
        '--sr', help='Sampling rate to be used', default=24000, type=int)
    parser.add_argument(
        'destfolder', help='Destination folder for processed files')
    parser.add_argument(
        'files', help='List of files to be processed', nargs='+')
    parser.add_argument(
        '--force', help='Forces preprocessign over all files', action='store_true')
    return parser.parse_args(args)


def save_audio(y, sr, destfolder, fname):
    soundfile.write(os.path.join(destfolder, fname), y, sr, subtype='PCM_24')


def save_descriptor(descriptor, destfolder, fname):
    path = os.path.join(destfolder, fname)
    md = media_descriptor.MediaDescriptor(path)
    md.update(descriptor)


def load_descriptor(destfolder, fname):
    path = os.path.join(destfolder, fname)
    md = media_descriptor.MediaDescriptor(path)
    return md.read()


def check_colision(destfolder, signature, raw_filepath):
    path = os.path.join(destfolder, signature + '.json')
    if os.path.exists(path):
        descriptor = load_descriptor(destfolder, signature + '.json')
        assert descriptor['raw_filepath'] == raw_filepath


def load_descriptors(destfolder, keep_fields):
    descriptors = {}
    for root, dirs, files in os.walk(destfolder):
        for f in files:
            if not '.json' in f:
                continue

            d = load_descriptor(destfolder, f)

            raw_filepath = d['raw_filepath']
            if not raw_filepath in descriptors.keys():
                descriptors[raw_filepath] = []

            signature = d['signature']
            keys = d.keys()
            for k in list(keys):
                if k not in keep_fields:
                    d.pop(k)

            descriptors[raw_filepath].append((d, signature))
    return descriptors


def search_for_preprocessing(descriptors, raw_filepath, **kargs):
    # print(descriptors)
    # print(raw_filepath)
    # print(kargs)
    if raw_filepath not in descriptors.keys():
        return None

    for entry, signature in descriptors[raw_filepath]:
        if kargs == entry:
            return signature

    return None


def preprocess_audio(path, **params):
    sr_target = params['sr']
    path = os.path.realpath(path)
    y, sr_original = librosa.load(path, sr=None)
    duration = librosa.get_duration(filename=path)

    # additional processing may be performed here

    y = librosa.resample(y, sr_original, sr_target,
                         res_type='linear', scale=True)

    output = media_audio.PreprocessedAudio(
        y, sr_target, duration, utils.make_path_relative(path))
    return output

def preprocess_file(path, descriptors, destfolder, sr):
    p_relative = utils.make_path_relative(path)
    signature = search_for_preprocessing(descriptors, p_relative, sr=sr)
    if signature is not None:
        print('Skiping processing for file "%s". Signature=%s' %
                (p_relative, signature))
        return

    preproc = preprocess_audio(p, sr=sr)
    signature = preproc.get_signature()

    check_colision(destfolder, signature, preproc.raw_filepath)
    save_audio(preproc.y, preproc.sr, destfolder, signature + '.wav')
    save_descriptor(preproc.get_descriptor(),
                    destfolder, signature + '.json')

if __name__ == '__main__':

    args = parse_args()

    destfolder = args.destfolder
    files = args.files
    sr = args.sr

    if not os.path.exists(destfolder):
        os.makedirs(destfolder)

    descriptors = {}
    if not args.force:
        descriptors = load_descriptors(destfolder, ['sr'])

    for p in files:
        preprocess_file(p, descriptors, destfolder, sr)