#! -*- coding: utf-8 -*-
# 词级别的中文RoFormer预训练
# MLM任务

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from tensorflow.keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay
from bert4keras.optimizers import extend_with_piecewise_linear_lr
from bert4keras.optimizers import extend_with_gradient_accumulation
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator
from bert4keras.snippets import text_segmentate
import jieba
import matplotlib.pyplot as plt
import pandas as pd

jieba.initialize()

from datasets import Dataset
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("Saving to:", os.getcwd())
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# Optional: Set GPU memory growth
physical_gpus = tf.config.list_physical_devices('GPU')
for gpu in physical_gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



# Use MultiWorker strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# 基本参数
maxlen = 512
batch_size = 64
epochs = 100

# bert配置
config_path = 'chinese_wobert_plus_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_wobert_plus_L-12_H-768_A-12/vocab.txt'

def corpus():
    file_path = "kaggle_dataset/train/dataset.arrow"
    ds = Dataset.from_file(file_path)
    for l in ds:
        for text in text_process(l['text']):
            yield text

def text_process(text):
    texts = text_segmentate(text, 32, u'\n。')
    result, length = '', 0
    for text in texts:
        if result and len(result) + len(text) > maxlen * 1.5:
            yield result
            result, length = '', 0
        result += text
    if result:
        yield result

tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

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

class data_generator(DataGenerator):
    def __iter__(self, random=False):
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            source, target = random_masking(token_ids)
            yield source, segment_ids, target

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


def sparsemax(logits, axis=-1):
    # 1) Stabilize
    logits -= tf.reduce_max(logits, axis=axis, keepdims=True)

    # 2) Sort
    z_sorted = tf.sort(logits, axis=axis, direction='DESCENDING')

    # 3) Cumsum & range
    z_cumsum = tf.cumsum(z_sorted, axis=axis)
    dim = tf.shape(logits)[axis]
    dim_float = tf.cast(dim, logits.dtype)
    
    # Build k = [1,2,…,dim] with shape [1,…,1, dim, 1,…,1]
    rank = logits.shape.rank or tf.rank(logits)
    # Create a shape of all ones, but size ‘dim’ at the projection axis:
    k_shape = [1] * rank
    k_shape[axis] = dim
    # Now build and reshape:
    k = tf.reshape(tf.range(1, dim_float + 1, dtype=logits.dtype), k_shape)

    # 4) Threshold condition
    z_check = 1 + k * z_sorted > z_cumsum
    k_z = tf.reduce_sum(tf.cast(z_check, tf.int32), axis=axis, keepdims=True)

    # 5) Compute tau
    # (Use same flat-index pattern you already have—now k_z has correct shape)
    batch_size = tf.shape(k_z)[0]
    k_z_flat = tf.reshape(k_z - 1, [batch_size])    # shape [batch]
    batch_idx = tf.range(batch_size, dtype=tf.int32) # shape [batch]
    indices = tf.stack([batch_idx, k_z_flat], axis=1)

    tau_sum = tf.gather_nd(
        tf.reshape(z_cumsum, [batch_size, dim]),     # reshape z_cumsum to 2D: [batch, dim]
        indices                                      # gathering along the class dim
    )
    tau = tf.reshape((tau_sum - 1) / tf.cast(k_z_flat + 1, logits.dtype),
                     [batch_size, 1] + [1] * (rank - 2))

    # 6) Final projection
    return tf.maximum(0., logits - tau)


class SparseMaxLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply sparsemax to logits
        y_pred = sparsemax(y_pred_logits)
        
        # Compute accuracy (similar to the original implementation)
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy', aggregation='mean')
        
        # Process in smaller batches to save memory
        y_true_flat = K.flatten(y_true)
        batch_range = K.arange(0, K.shape(y_true_flat)[0])
        indices = K.stack([batch_range, K.cast(y_true_flat, 'int32')], axis=1)
        
        # Extract logit for the true class (memory efficient)
        z_y = tf.gather_nd(K.reshape(y_pred_logits, [-1, K.shape(y_pred_logits)[-1]]), indices)
        
        # Compute squared norm more efficiently
        squared_norm = K.sum(K.square(y_pred), axis=-1)
        
        # Compute loss
        loss = -z_y + 0.5 * K.reshape(squared_norm, K.shape(y_true_flat)) + 0.5
        loss = K.reshape(loss, K.shape(y_true))
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
    # outputs = CrossEntropy(1)([y_in, model.output])
    outputs = SparseMaxLoss(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
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

    history = train_model.fit(
        dataset, steps_per_epoch=1000, epochs=epochs, callbacks=[evaluator]
    )

    # Plot training accuracy and loss
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='orange')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    #plt.savefig('Training_Accuracy_and_Loss_Curve.png')
    plt.savefig('/workspace/Quant/Sparsemax_Training_Accuracy_and_Loss_Curve.png')
    plt.show()
    print("Saving to:", os.getcwd())
    # Add after training
    print("History keys:", history.history.keys())
    print("Accuracy values:", history.history.get('accuracy', []))
    print("Loss values:", history.history.get('loss', []))
    metrics_df = pd.DataFrame(history.history)
    # Save to CSV file inside your host-mounted folder
    metrics_df.to_csv('/workspace/Quant/training_metrics.csv', index=False)

else:
    model.load_weights('bert_model.weights')
