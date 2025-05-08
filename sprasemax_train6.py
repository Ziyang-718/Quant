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

# 基本参数
maxlen = 512
batch_size = 32  # Reduced to improve stability
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

def safe_sparsemax(logits, axis=-1):
    """
    Improved sparsemax implementation with better numerical stability.
    Modified to handle edge cases and maintain numerical stability.
    
    Args:
        logits: Input tensor
        axis: Dimension along which to apply sparsemax
        
    Returns:
        sparsemax activations
    """
    epsilon = 1e-10
    
    # Initial preprocessing for numerical stability
    # This ensures we don't overflow or get undefined behavior
    logits = tf.cast(logits, tf.float32)
    logits_max = tf.reduce_max(logits, axis=axis, keepdims=True)
    logits_shifted = logits - logits_max
    
    # Flatten for easier processing
    original_shape = tf.shape(logits)
    if axis != -1:
        # Not implemented for arbitrary axes yet
        axis = axis if axis >= 0 else tf.rank(logits) + axis
    
    # Reshape to 2D [batch_size, dim_size]
    dim_size = original_shape[-1]
    logits_2d = tf.reshape(logits_shifted, [-1, dim_size])
    
    # Sort in descending order for projection algorithm
    z_sorted = tf.sort(logits_2d, axis=-1, direction='DESCENDING')
    
    # Compute cumulative sum for the projection
    z_cumsum = tf.cumsum(z_sorted, axis=-1)
    
    # Create range tensor [1, 2, ..., dim_size]
    k_tensor = tf.range(1, dim_size + 1, dtype=tf.float32)
    k_tensor = tf.reshape(k_tensor, [1, -1])  # [1, dim_size]
    
    # Calculate projection threshold: z_i - (sum_j z_j - 1) / k > 0
    # Equivalent to: k * z_i > cumsum_i - 1
    condition = k_tensor * z_sorted > z_cumsum - 1.0
    
    # Count elements that satisfy the condition (potential non-zero elements)
    k_values = tf.reduce_sum(tf.cast(condition, tf.float32), axis=-1)
    
    # Ensure k is at least 1 (always at least one non-zero element)
    k_values = tf.maximum(k_values, 1.0)
    
    # Get the corresponding k for each row
    k_indices = tf.cast(k_values, tf.int32) - 1  # Convert to 0-based indexing
    
    # Create row indices for gathering from the cumsum
    row_indices = tf.range(tf.shape(logits_2d)[0], dtype=tf.int32)
    gather_indices = tf.stack([row_indices, k_indices], axis=1)
    
    # Get cumulative sum at position k for each row
    tau_sum = tf.gather_nd(z_cumsum, gather_indices)
    
    # Calculate the threshold (tau)
    tau = (tau_sum - 1.0) / k_values
    tau = tf.reshape(tau, [-1, 1])  # [batch_size, 1] for broadcasting
    
    # Calculate the output (sparsemax)
    # max(0, x_i - tau)
    sparse_out = tf.maximum(0.0, logits_2d - tau)
    
    # Normalize to ensure sum = 1 (just in case)
    sum_sparse = tf.reduce_sum(sparse_out, axis=-1, keepdims=True)
    sum_sparse = tf.maximum(sum_sparse, epsilon)  # Avoid division by zero
    normalized_out = sparse_out / sum_sparse
    
    # Reshape back to original shape
    result = tf.reshape(normalized_out, original_shape)
    
    return result

class ImprovedSparseMaxLoss(Loss):
    """
    Improved Sparsemax loss implementation with better tracking of metrics.
    This implementation is specifically designed to avoid NaN issues.
    """
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-10
        
        # Apply padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Compute sparsemax probabilities carefully
        y_pred = safe_sparsemax(y_pred_logits)
        
        # Get vocabulary size
        vocab_size = tf.shape(y_pred_logits)[-1]
        
        # Convert true labels to one-hot with careful handling of invalid indices
        # Ensure indices are valid (between 0 and vocab_size-1)
        y_true_clipped = tf.clip_by_value(
            tf.cast(y_true, tf.int32),
            0,
            vocab_size - 1
        )
        y_true_oh = tf.one_hot(y_true_clipped, vocab_size, dtype=tf.float32)
        
        # Calculate sparsemax loss components
        
        # First term: -<y_true, logits>
        # Clip logits to avoid extreme values
        logits_clipped = tf.clip_by_value(y_pred_logits, -1e6, 1e6)
        log_term = -tf.reduce_sum(y_true_oh * logits_clipped, axis=-1)
        
        # Second term: 0.5 * ||y_pred||^2
        squared_norm = 0.5 * tf.reduce_sum(tf.square(y_pred), axis=-1)
        
        # Constant term
        constant_term = 0.5
        
        # Calculate point-wise loss
        point_loss = log_term + squared_norm + constant_term
        
        # Apply mask and calculate mean
        masked_loss = point_loss * y_mask
        mean_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(y_mask) + epsilon)
        
        # Calculate and track accuracy with improved handling
        # Using sparse categorical accuracy directly on sparsemax outputs
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy_val = tf.reduce_sum(accuracy * y_mask) / (tf.reduce_sum(y_mask) + epsilon)
        self.add_metric(accuracy_val, name='accuracy', aggregation='mean')
        
        # Add additional monitoring metrics
        # Track the actual values to help diagnose issues
        self.add_metric(
            tf.reduce_mean(log_term), 
            name='log_term_mean', 
            aggregation='mean'
        )
        self.add_metric(
            tf.reduce_mean(squared_norm), 
            name='squared_norm_mean', 
            aggregation='mean'
        )
        
        return mean_loss

class SafeEvaluator(keras.callbacks.Callback):
    """Improved evaluator with better handling of metrics."""
    
    def __init__(self, save_interval=1, model_path='bert_model.weights'):
        super(SafeEvaluator, self).__init__()
        self.save_interval = save_interval
        self.model_path = model_path
        self.best_loss = float('inf')
        self.metrics_history = {
            'loss': [],
            'accuracy': []
        }
    
    def on_batch_end(self, batch, logs=None):
        # Monitor early batches closely
        if batch < 5:
            loss_val = logs.get('loss', float('nan'))
            acc_val = logs.get('accuracy', 0.0)
            print(f"Batch {batch}: loss={loss_val:.6f}, acc={acc_val:.6f}")
            
            # Check for concerning values
            if np.isnan(loss_val) or np.isinf(loss_val):
                print(f"WARNING: Detected bad loss value: {loss_val}")
    
    def on_epoch_end(self, epoch, logs=None):
        # Safe handling of metrics
        loss_val = logs.get('loss', float('nan'))
        acc_val = logs.get('accuracy', 0.0)
        log_term = logs.get('log_term_mean', float('nan'))
        squared_norm = logs.get('squared_norm_mean', float('nan'))
        
        # Store metrics
        self.metrics_history['loss'].append(loss_val)
        self.metrics_history['accuracy'].append(acc_val)
        
        # Save weights periodically
        if (epoch + 1) % self.save_interval == 0:
            self.model.save_weights(self.model_path)
            print(f"Model saved at epoch {epoch+1}")
        
        # Save best model
        if loss_val < self.best_loss and not np.isnan(loss_val):
            self.best_loss = loss_val
            self.model.save_weights(f"{self.model_path}.best")
            print(f"New best model saved with loss: {loss_val:.6f}")
        
        # Print detailed status
        print(f"\nEpoch {epoch+1}: loss={loss_val:.6f}, acc={acc_val:.6f}")
        print(f"Components: log_term={log_term:.6f}, squared_norm={squared_norm:.6f}")
        
        # Save metrics to CSV after each epoch
        df = pd.DataFrame(self.metrics_history)
        df.to_csv('/workspace/Quant/training_metrics_progress.csv', index=False)

# Choose which strategy to use
# Try with simpler single-GPU first if having issues
USE_MULTI_GPU = True

if USE_MULTI_GPU:
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
else:
    # Fallback to simple strategy if multi-worker has issues
    strategy = tf.distribute.get_strategy()

with strategy.scope():
    # Build model with appropriate options
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
    
    # Use improved sparsemax loss
    outputs = ImprovedSparseMaxLoss(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    # Configure optimizer with conservative settings
    optimizer = Adam(
        learning_rate=1e-5,  # Lower initial learning rate
        epsilon=1e-8,
        clipnorm=1.0,  # Add gradient clipping
        clipvalue=5.0   # More conservative clipping
    )
    
    # Compile with more careful options
    train_model.compile(
        optimizer=optimizer,
        loss=None  # Loss is computed in the Loss layer
    )
    
    train_model.summary()
    # Load pre-trained weights
    bert.load_weights_from_checkpoint(checkpoint_path)

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    
    # Initialize evaluator with more frequent saving
    evaluator = SafeEvaluator(save_interval=1)
    
    # Add terminate_on_nan for safety
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    # Add learning rate scheduler
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Add model checkpoint callback as backup
    checkpoint = keras.callbacks.ModelCheckpoint(
        'bert_model.weights.{epoch:02d}',
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )
    
    # Prepare data generator with smaller batch size
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )
    
    # Use prefetch for better performance
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    try:
        # Train with improved monitoring
        history = train_model.fit(
            dataset, 
            steps_per_epoch=200,  # Reduced for faster epochs
            epochs=epochs, 
            callbacks=[
                evaluator,
                terminate_on_nan,
                lr_scheduler,
                checkpoint
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
        
        # Save detailed metrics to CSV
        metrics_df = pd.DataFrame(history.history)
        metrics_df.to_csv('/workspace/Quant/sparsemax_training_metrics.csv', index=False)
    
    except Exception as e:
        print(f"Training error: {e}")
        # Save model on error
        model.save_weights('bert_model.weights.error')
else:
    model.load_weights('bert_model.weights')

