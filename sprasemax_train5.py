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

def basic_sparsemax(logits):
    """Extremely simplified sparsemax implementation suitable for MLM pretraining.
    This avoids complex reshape operations that can cause issues in distributed training.
    
    Args:
        logits: Input tensor with shape [batch_size, sequence_length, vocab_size]
        
    Returns:
        Sparsemax probabilities with same shape as input
    """
    # Keep standard operations that are well-tested
    epsilon = 1e-10
    
    # For numerical stability - like in softmax
    logits_shifted = logits - tf.reduce_max(logits, axis=-1, keepdims=True)
    
    # Get original shape
    shape = tf.shape(logits)
    
    # Flatten all but the last dimension for easier processing
    logits_2d = tf.reshape(logits_shifted, [-1, shape[-1]])
    
    # Sort values in descending order
    z_sorted = tf.sort(logits_2d, axis=-1, direction='DESCENDING')
    
    # Compute cumulative sum
    z_cumsum = tf.cumsum(z_sorted, axis=-1)
    
    # Get size of vocabulary dimension
    vocab_size = shape[-1]
    
    # Create range tensor [1, 2, 3, ..., vocab_size]
    k = tf.range(1, vocab_size + 1, dtype=tf.float32)
    k = tf.expand_dims(k, axis=0)  # [1, vocab_size]
    
    # Compute threshold indices
    threshold_condition = 1.0 + k * z_sorted > z_cumsum
    
    # Count how many values satisfy the condition for each row
    # This will give us the k value for each example
    k_values = tf.reduce_sum(tf.cast(threshold_condition, tf.float32), axis=-1)
    
    # Handle edge case: if k is 0, set it to 1
    k_values = tf.maximum(k_values, 1.0)
    
    # Create row indices
    row_indices = tf.range(tf.shape(logits_2d)[0], dtype=tf.int32)
    
    # Convert k_values to indices (0-based)
    k_indices = tf.cast(k_values, tf.int32) - 1
    
    # Create gather indices
    gather_indices = tf.stack([row_indices, k_indices], axis=1)
    
    # Gather the cumulative sum values at these indices
    # This gives us the sum of all elements that should be non-zero
    cumsum_values = tf.gather_nd(z_cumsum, gather_indices)
    
    # Compute tau (threshold)
    # tau = (cumsum_k - 1) / k
    taus = (cumsum_values - 1.0) / k_values
    taus = tf.expand_dims(taus, axis=-1)  # Make broadcastable with logits
    
    # Apply the threshold: max(0, x - tau)
    sparse_outputs = tf.maximum(0.0, logits_2d - taus)
    
    # Normalize to ensure sum is 1 (important for a probability distribution)
    sum_values = tf.reduce_sum(sparse_outputs, axis=-1, keepdims=True)
    sum_values = tf.maximum(sum_values, epsilon)  # Avoid division by zero
    sparse_outputs = sparse_outputs / sum_values
    
    # Reshape back to original shape
    outputs = tf.reshape(sparse_outputs, shape)
    
    return outputs

class SimpleSparseMaxLoss(Loss):
    """Simplified sparsemax loss implementation focused on stability and performance."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-10
        
        # Create padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply simplified sparsemax
        y_pred = basic_sparsemax(y_pred_logits)
        
        # Calculate and track accuracy
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        valid_count = tf.reduce_sum(y_mask) + epsilon
        accuracy_val = tf.reduce_sum(accuracy * y_mask) / valid_count
        self.add_metric(accuracy_val, name='accuracy', aggregation='mean')
        
        # Get shape information
        vocab_size = tf.shape(y_pred_logits)[-1]
        
        # Convert true labels to one-hot encoding
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
        
        # Compute the standard sparsemax loss
        # First term: -<y_true, logits>
        log_term = -tf.reduce_sum(y_true_oh * y_pred_logits, axis=-1)
        
        # Second term: 0.5 * ||y_pred||^2
        squared_norm = 0.5 * tf.reduce_sum(tf.square(y_pred), axis=-1)
        
        # Constant term
        constant_term = 0.5
        
        # Full loss
        point_loss = log_term + squared_norm + constant_term
        
        # Apply mask
        masked_loss = point_loss * y_mask
        
        # Calculate mean loss
        mean_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(y_mask) + epsilon)
        
        # Detect and handle NaN/Inf
        loss_is_bad = tf.logical_or(
            tf.math.is_nan(mean_loss),
            tf.math.is_inf(mean_loss)
        )
        safe_loss = tf.where(
            loss_is_bad,
            tf.constant(0.1, dtype=tf.float32),  # Fallback value if NaN/Inf detected
            mean_loss
        )
        
        return safe_loss

class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('bert_model.weights')
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
    
    # Use simplified SparseMax loss
    outputs = SimpleSparseMaxLoss(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    
    # Configure optimizer with better parameters
    optimizer = AdamWLRG(
        learning_rate=5e-5,  # Increased learning rate
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={20000: 0.5, 40000: 0.1},  # Step schedule
        epsilon=1e-8,  # Improved numerical stability
        clipnorm=1.0,  # Gradient clipping
        clipvalue=10.0  # Value clipping
    )
    
    # Compile with minimal options
    train_model.compile(optimizer=optimizer)
    
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    evaluator = Evaluator()
    
    # Minimal callbacks to reduce overhead
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    # Prepare data generator
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )

    # Train with minimal callbacks
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
    plt.savefig('/workspace/Quant/Sparsemax_Training_Accuracy_and_Loss_Curve.png')
    plt.show()
    
    print("Saving to:", os.getcwd())
    print("History keys:", history.history.keys())
    print("Accuracy values:", history.history.get('accuracy', []))
    print("Loss values:", history.history.get('loss', []))
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(history.history)
    metrics_df.to_csv('/workspace/Quant/Sparsemax_training_metrics.csv', index=False)
else:
    model.load_weights('bert_model.weights')