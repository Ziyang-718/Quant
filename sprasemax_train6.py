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

def project_simplex(v, z=1.0):
    """
    TensorFlow implementation of the simplex projection operation.
    Implementation follows the algorithm from the sparsemax paper.
    
    Args:
        v: Input tensor of shape [..., n]
        z: Simplex parameter (default: 1.0)
    
    Returns:
        Projected tensor of same shape as v
    """
    # Sort v in descending order
    v_sorted = tf.sort(v, axis=-1, direction='DESCENDING')
    
    # Compute cumulative sum
    cumsum = tf.cumsum(v_sorted, axis=-1)
    
    # Get the size of the last dimension
    dim = tf.shape(v)[-1]
    
    # Create range tensor [1, 2, ..., dim]
    range_tensor = tf.range(1, dim + 1, dtype=tf.float32)
    range_tensor = tf.cast(range_tensor, v.dtype)
    
    # Reshape for broadcasting
    range_tensor = tf.reshape(range_tensor, [1] * (len(v.shape) - 1) + [dim])
    
    # Compute the threshold condition: v_sorted_i > (cumsum_i - z) / i
    condition = v_sorted > (cumsum - z) / range_tensor
    
    # Find the largest k that satisfies the condition
    # We're looking for the highest index where condition is True
    # First, convert condition to int and then use sum
    k = tf.reduce_sum(tf.cast(condition, tf.int32), axis=-1, keepdims=True) - 1
    
    # Calculate the threshold tau
    # We need to gather the cumsum at index k for each row
    # First, create indices for batch_gather
    batch_shape = tf.shape(v)[:-1]
    batch_size = tf.reduce_prod(batch_shape)
    batch_indices = tf.reshape(tf.range(batch_size), batch_shape)
    
    # Reshape k to match batch_indices
    k_reshaped = tf.reshape(k, batch_shape)
    
    # Stack indices for gather_nd
    indices = tf.stack([
        tf.reshape(batch_indices, [-1]),
        tf.reshape(k_reshaped, [-1])
    ], axis=1)
    
    # Reshape cumsum to 2D for gather_nd
    cumsum_2d = tf.reshape(cumsum, [batch_size, dim])
    
    # Gather the cumulative sum at position k
    cumsum_k = tf.gather_nd(cumsum_2d, indices)
    
    # Reshape back to match batch_shape
    cumsum_k = tf.reshape(cumsum_k, batch_shape)
    
    # Reshape k for division
    k_float = tf.cast(k, v.dtype)
    k_float = tf.reshape(k_float, batch_shape)
    
    # Calculate tau
    tau = (cumsum_k - z) / (k_float + 1.0)
    
    # Reshape tau for broadcasting
    tau = tf.expand_dims(tau, axis=-1)
    
    # Apply the projection: max(0, v - tau)
    w = tf.maximum(0.0, v - tau)
    
    return w

def sparsemax_loss(y_true, y_pred, y_pred_logits):
    """
    Compute the sparsemax loss as described in the original paper.
    
    Args:
        y_true: One-hot encoded target
        y_pred: Sparsemax output (probabilities)
        y_pred_logits: Logits before sparsemax
    
    Returns:
        Sparsemax loss
    """
    # Compute the dot product term
    support = y_pred > 0
    support_float = tf.cast(support, tf.float32)
    q_dot_z = tf.reduce_sum(y_true * y_pred_logits, axis=-1)
    
    # Compute the squared norm term
    squared_norm = tf.reduce_sum(tf.square(y_pred), axis=-1)
    
    # Put it all together
    loss = -q_dot_z + 0.5 * squared_norm + 0.5
    
    return loss

class SparsemaxLoss(Loss):
    """Improved Sparsemax loss function based on the algorithm from the original paper."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-12
        
        # Create padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply simplex projection to compute sparsemax
        # First normalize logits for numerical stability
        logits_shifted = y_pred_logits - tf.reduce_max(y_pred_logits, axis=-1, keepdims=True)
        y_pred = project_simplex(logits_shifted)
        
        # Compute accuracy
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        valid_count = tf.reduce_sum(y_mask) + epsilon
        accuracy_val = tf.reduce_sum(accuracy * y_mask) / valid_count
        self.add_metric(accuracy_val, name='accuracy', aggregation='mean')
        
        # Convert true labels to one-hot
        y_true_int = tf.cast(y_true, tf.int32)
        vocab_size = tf.shape(y_pred_logits)[-1]
        y_true_oh = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
        
        # Compute sparsemax loss
        point_loss = sparsemax_loss(y_true_oh, y_pred, logits_shifted)
        
        # Apply mask
        masked_loss = point_loss * y_mask
        
        # Calculate mean loss
        mean_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(y_mask) + epsilon)
        
        return mean_loss

class MLMLossChoice(Loss):
    """Loss function that can switch between Sparsemax and CrossEntropy"""
    
    def __init__(self, use_sparsemax=True, **kwargs):
        super(MLMLossChoice, self).__init__(**kwargs)
        self.use_sparsemax = use_sparsemax
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-12
        
        # Create padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        if self.use_sparsemax:
            # Use sparsemax
            logits_shifted = y_pred_logits - tf.reduce_max(y_pred_logits, axis=-1, keepdims=True)
            y_pred = project_simplex(logits_shifted)
            
            # Convert to one-hot for sparsemax loss
            y_true_int = tf.cast(y_true, tf.int32)
            vocab_size = tf.shape(y_pred_logits)[-1]
            y_true_oh = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
            
            # Compute loss
            point_loss = sparsemax_loss(y_true_oh, y_pred, logits_shifted)
        else:
            # Use standard softmax + cross-entropy
            y_pred = tf.nn.softmax(y_pred_logits, axis=-1)
            point_loss = K.sparse_categorical_crossentropy(
                y_true, y_pred_logits, from_logits=True
            )
        
        # Track accuracy
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        valid_count = tf.reduce_sum(y_mask) + epsilon
        accuracy_val = tf.reduce_sum(accuracy * y_mask) / valid_count
        self.add_metric(accuracy_val, name='accuracy', aggregation='mean')
        
        # Apply mask and compute mean
        masked_loss = point_loss * y_mask
        mean_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(y_mask) + epsilon)
        
        return mean_loss

class Evaluator(keras.callbacks.Callback):
    def __init__(self, save_interval=1):
        super(Evaluator, self).__init__()
        self.save_interval = save_interval
    
    def on_epoch_end(self, epoch, logs=None):
        # Save weights at specified intervals
        if (epoch + 1) % self.save_interval == 0:
            model.save_weights('bert_model.weights')
        
        # Print status
        print(f"\nEpoch {epoch+1}: loss={logs.get('loss'):.6f}, acc={logs.get('accuracy'):.6f}")

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
    
    # Use the flexible loss function - you can toggle between sparsemax and cross-entropy
    outputs = MLMLossChoice(use_sparsemax=True, output_axis=1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    
    # Configure optimizer
    optimizer = AdamWLRG(
        learning_rate=2e-5,  # Standard learning rate for BERT
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={20000: 0.5, 40000: 0.1},
    )
    
    # Compile
    train_model.compile(optimizer=optimizer)
    
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    
    # Initialize evaluator
    evaluator = Evaluator(save_interval=1)
    
    # Add terminate on NaN callback
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    # Prepare data generator
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )
    
    # Use prefetch for better performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    # Train
    history = train_model.fit(
        dataset, 
        steps_per_epoch=1000, 
        epochs=epochs, 
        callbacks=[
            evaluator,
            terminate_on_nan
        ],
        verbose=1
    )

    # Plot training metrics
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
    plt.savefig('/workspace/Quant/Training_Accuracy_and_Loss_Curve.png')
    plt.show()
    
    print("Saving to:", os.getcwd())
    print("History keys:", history.history.keys())
    print("Accuracy values:", history.history.get('accuracy', []))
    print("Loss values:", history.history.get('loss', []))
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(history.history)
    metrics_df.to_csv('/workspace/Quant/training_metrics.csv', index=False)
else:
    model.load_weights('bert_model.weights')
