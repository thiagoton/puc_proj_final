import keras
import generator
import numpy as np
import tensorflow as tf

class EvaluatorBase:
    def __init__(self) -> None:
        pass

    def evaluate(model: keras.Model, validation_list: list, **kwargs):
        raise NotImplementedError

class MajorityVotingEvaluator(EvaluatorBase):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def get_max_votes(one_hot_pred: np.array):
        votes = np.argmax(one_hot_pred, axis=-1)
        labels, counts = np.unique(votes, return_counts=True)
        max_id = np.argmax(counts)

        return labels[max_id], counts[max_id]

    def evaluate(self, model: keras.Model, validation_list: list, **kwargs):
        model_input = kwargs['model_input']
        window_overlap = kwargs['window_overlap']
        gen = generator.DataGenerator(validation_list, 1, model_input, window_overlap)

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

        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        metric.update_state(labels_true, labels_pred)
        acc = metric.result().numpy()
        confusion = tf.math.confusion_matrix(np.squeeze(labels_true), np.squeeze(labels_pred))
        return acc, tf.constant(confusion).numpy()

model_path = 'experiments/2021-12-31_13-21-45/checkpoints/ckpt_00031-00000-00000.h5'
model = keras.models.load_model(model_path)

validationlist = []
with open('datasets/testlist.txt', 'r') as f:
    for line in f.readlines():
        validationlist.append(line.replace('\n', ''))

evaluator = MajorityVotingEvaluator()
acc, confusion = evaluator.evaluate(model, validationlist, model_input=int(1.5*24000), window_overlap=0.5)

print(confusion/(confusion.astype(np.float).sum(axis=1)))
print(acc)