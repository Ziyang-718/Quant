

import os
import json
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd

from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import extend_with_weight_decay, extend_with_piecewise_linear_lr, extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import text_segmentate
from datasets import Dataset
import jieba

# Initialize jieba
jieba.initialize()

# Environment settings
os.environ['TF_KERAS'] = '1'  # Use tf.keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["SM_FRAMEWORK"] = "tf.keras"

print("Saving to:", os.getcwd())

# GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# MultiWorker strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# Basic parameters
maxlen = 512
batch_size = 64
epochs = 100

# Bert paths
config_path = 'chinese_wobert_plus_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_wobert_plus_L-12_H-768_A-12/vocab.txt'

# Dataset
def corpus():
    file_path = "kaggle_dataset/train/dataset.arrow"
    ds = Dataset.from_file(file_path)
    for l in ds:
        for text in text_process(l['text']):
            yield text

def text_process(text):
    texts = text_segmentate(text, 32, u'\n。')
    result = ''
    for text in texts:
        if result and len(result) + len(text) > maxlen * 1.5:
            yield result
            result = ''
        result += text
    if result:
        yield result

# Tokenizer
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

# Random masking for MLM
def random_masking(token_ids):
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target

# Data generator
class data_generator(DataGenerator):
    def __iter__(self, random=False):
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            source, target = random_masking(token_ids)
            yield source, segment_ids, target

# Loss function
class CrossEntropy(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy', aggregation='mean')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

with strategy.scope():
    bert = build_transformer_model(
        config_path,
        checkpoint_path=None,
        model='roformer',
        with_mlm='linear',
        ignore_invalid_weights=True,
        return_keras_model=False
    )
    model = bert.model

    y_in = keras.layers.Input(shape=(None,), name='Input-Label')
    outputs = CrossEntropy(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(keras.optimizers.Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    optimizer = AdamWLRG(
        learning_rate=1e-5,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={20000: 1}
    )
    train_model.compile(optimizer=optimizer)
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)

    # Model Parameters and Size Info
    trainable_params = np.sum([K.count_params(w) for w in train_model.trainable_weights])
    non_trainable_params = np.sum([K.count_params(w) for w in train_model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    size_in_bytes = total_params * 4  # float32 = 4 bytes
    size_in_mb = size_in_bytes / (1024 ** 2)
    print(f"Estimated model size: {size_in_mb:.2f} MB")

# Evaluator callback
class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('bert_model.weights')

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))

    evaluator = Evaluator()
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )

    # --- Training start time ---
    start_time = time.time()

    history = train_model.fit(
        dataset, steps_per_epoch=1000, epochs=epochs, callbacks=[evaluator]
    )

    # --- Training end time ---
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")

    # Save training time
    with open('/workspace/Quant/training_time.txt', 'w') as f:
        f.write(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)\n")

    # Plot training accuracy and loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/workspace/Quant/Training_Accuracy_and_Loss_Curve.png')
    plt.show()

    print("Saving to:", os.getcwd())
    print("History keys:", history.history.keys())
    print("Accuracy values:", history.history.get('accuracy', []))
    print("Loss values:", history.history.get('loss', []))

    metrics_df = pd.DataFrame(history.history)
    metrics_df.to_csv('/workspace/Quant/training_metrics.csv', index=False)

else:
    model.load_weights('bert_model.weights')
