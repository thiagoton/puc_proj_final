import tensorflow as tf
import generator
import numpy as np
import tensorflow as tf
import argparse
import sys
import os
import sklearn
import model
import json

# import infrastructure
from common import utils  # nopep8
from common import metadata_extractor  # nopep8


def confusion_matrix(preds: np.array, trues: np.array, normalize=False):
    '''
    Computes the confusion matrix
    @param preds: predictions
    @param trues: expected true labels
    @param normalize:   True - normalizes confusion matrix
                        False - raw confusion values
    @return confusion matrix
    '''
    tf_confusion = tf.math.confusion_matrix(
        np.squeeze(trues), np.squeeze(preds))
    confusion = tf.constant(tf_confusion).numpy()
    if (normalize):
        confusion = confusion/(confusion.astype(np.float).sum(axis=1))
    return tf.constant(confusion).numpy()


class EvaluatorBase:
    def __init__(self) -> None:
        pass

    def evaluate(model: tf.keras.Model, validation_list: list, **kwargs) -> dict:
        raise NotImplementedError


class MajorityVotingEvaluator(EvaluatorBase):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.vote_th = kwargs.get('vote_th', 0.8)

    def get_max_votes(self, one_hot_pred: np.array):
        one_hot_pred[one_hot_pred > self.vote_th] = 1
        one_hot_pred[one_hot_pred <= self.vote_th] = 0
        sum_one_hot = np.sum(one_hot_pred, axis=-1)

        non_noise_one_hot = one_hot_pred[sum_one_hot > 0]
        votes_noise = np.sum(sum_one_hot == 0)

        votes = np.argmax(non_noise_one_hot, axis=-1)
        labels, counts = np.unique(votes, return_counts=True)

        max_count = 0
        max_label = metadata_extractor.ALLOWED_CLASSES['noise']
        if len(counts):
            max_id = np.argmax(counts)
            max_count = counts[max_id]
            max_label = labels[max_id]

        if votes_noise > max_count:
            return metadata_extractor.ALLOWED_CLASSES['noise'], votes_noise

        return max_label, max_count

    def evaluate(self, model: tf.keras.Model, validation_list: list, **kwargs):
        model_input = kwargs['model_input']
        window_overlap = kwargs['window_overlap']
        gen = generator.DataGenerator(
            validation_list, 1, model_input, window_overlap)

        labels_true = None
        labels_pred = None
        for n in range(len(gen)):
            x, y_true = gen.get_batch_item(n, shuffle=False)
            y_pred = model.predict(x)
            label_true = MajorityVotingEvaluator.get_max_votes(y_true)[0]
            label_pred = MajorityVotingEvaluator.get_max_votes(y_pred)[0]

            if labels_true is None:
                labels_true = label_true
                labels_pred = label_pred
            else:
                labels_true = np.vstack((labels_true, label_true))
                labels_pred = np.vstack((labels_pred, label_pred))

        labels_pred = np.squeeze(labels_pred)
        labels_true = np.squeeze(labels_true)

        ordered_labels = metadata_extractor.get_labels(
            metadata_extractor.ALLOWED_CLASSES)
        output_dict = kwargs.get('output_dict', False)
        metrics = sklearn.metrics.classification_report(
            labels_true, labels_pred, target_names=ordered_labels, output_dict=output_dict, labels=[0,1,2])

        samples = validation_list
        preds = [ordered_labels[x] for x in labels_pred]
        true = [ordered_labels[x] for x in labels_true]
        return metrics, samples, preds, true


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluation script for trained models')
    parser.add_argument('model', help='path to keras model to be evaluated')
    parser.add_argument('params', help='yaml for parameters of provided file')
    parser.add_argument(
        'filelist', help='path to filelist to be used for evaluation')
    parser.add_argument('--output', help='Path to save outputs')

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    return args


def evaluate(args):
    model_path = args.model

    params = utils.load_params(args.params)
    validationlist = utils.load_filelist(args.filelist)

    factory = model.get_factory(params)
    m = factory.build_model()
    m.load_weights(model_path)

    evaluator = factory.get_evaluator(params['Evaluator'])

    output_dict = True if args.output is not None else False
    metrics, samples, pred, true = evaluator.evaluate(m,
                                                      validationlist,
                                                      model_input=factory.INPUT_SIZE,
                                                      window_overlap=params['window_overlap'],
                                                      output_dict=output_dict)

    if output_dict:
        out_dir = os.path.dirname(args.output)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(args.output, 'metrics.json')
        with open(path, 'w') as fd:
            json.dump(metrics, fd)
        path = os.path.join(args.output, 'result.csv')
        with open(path, 'w') as fd:
            fd.write('sample,pred,true\r\n')
            for n in range(len(samples)):
                fd.write('%s,%s,%s\r\n' %
                         (samples[n], pred[n], true[n]))
    else:
        print(metrics)
        print('sample,pred,true\r\n')
        for n in range(len(samples)):
            print('%s,%s,%s' % (samples[n], pred[n], true[n]))


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
