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


def efficient_sparsemax(logits, axis=-1):
    """Efficient sparsemax implementation with better numerical stability for training.
    
    This implementation combines efficiency with numerical stability to ensure
    proper gradient flow during training.
    """
    # For numerical stability
    epsilon = 1e-10
    
    # Cast inputs to float32
    logits = tf.cast(logits, tf.float32)
    
    # Get original shape
    original_shape = tf.shape(logits)
    
    # Get the last dimension size (vocabulary size)
    vocab_size = original_shape[axis]
    
    # Reshape to 2D for efficient processing
    if axis != -1 and axis != len(logits.shape) - 1:
        # If not last dimension, transpose to make it last
        perm = list(range(len(logits.shape)))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        logits = tf.transpose(logits, perm=perm)
    
    # Flatten to 2D: [batch_size, vocab_size]
    # Where batch_size may include other dimensions
    flattened_shape = [-1, original_shape[-1]]
    logits_2d = tf.reshape(logits, flattened_shape)
    
    # Subtract max for numerical stability (like softmax)
    logits_2d = logits_2d - tf.reduce_max(logits_2d, axis=-1, keepdims=True)
    
    # Sort in descending order
    sorted_logits = tf.sort(logits_2d, axis=-1, direction='DESCENDING')
    
    # Compute cumulative sum
    cumsum = tf.cumsum(sorted_logits, axis=-1)
    
    # Range tensor [1, 2, ..., vocab_size]
    k_tensor = tf.range(1, vocab_size + 1, dtype=tf.float32)
    k_tensor = tf.expand_dims(k_tensor, axis=0)  # Shape [1, vocab_size]
    
    # Compute threshold indices
    thresholds = 1.0 + k_tensor * sorted_logits
    valid_indices = tf.cast(thresholds > cumsum, tf.float32)
    
    # Find the number of valid indices (k)
    k = tf.reduce_sum(valid_indices, axis=-1)
    # Ensure k is at least 1
    k = tf.maximum(k, 1.0)
    
    # Compute threshold value tau
    # Get the position right before the threshold
    k_indices = tf.cast(k, tf.int32) - 1
    batch_indices = tf.range(tf.shape(logits_2d)[0])
    gather_indices = tf.stack([batch_indices, k_indices], axis=1)
    
    # Get the corresponding cumulative sum values
    cumsum_threshold = tf.gather_nd(cumsum, gather_indices)
    
    # Compute tau
    tau = (cumsum_threshold - 1.0) / k
    tau = tf.expand_dims(tau, axis=-1)  # Make it broadcastable
    
    # Apply the threshold
    sparse_output = tf.maximum(0.0, logits_2d - tau)
    
    # Normalize to ensure sum = 1
    sum_output = tf.reduce_sum(sparse_output, axis=-1, keepdims=True)
    # Avoid division by zero
    sum_output = tf.maximum(sum_output, epsilon)
    sparse_output = sparse_output / sum_output
    
    # Handle potential NaN/Inf values - fallback to softmax
    softmax_fallback = tf.nn.softmax(logits_2d)
    sparse_output = tf.where(
        tf.math.is_finite(sparse_output),
        sparse_output,
        softmax_fallback
    )
    
    # Reshape back to original shape
    result = tf.reshape(sparse_output, original_shape)
    
    # If we transposed earlier, transpose back
    if axis != -1 and axis != len(logits.shape) - 1:
        perm = list(range(len(logits.shape)))
        perm[axis], perm[-1] = perm[-1], perm[axis]
        result = tf.transpose(result, perm=perm)
    
    return result


class EnhancedSparseMaxLoss(Loss):
    """Enhanced SparseMax loss with improvements for better training stability and convergence."""
    
    def __init__(self, scale_factor=10.0, label_smoothing=0.1, **kwargs):
        super(EnhancedSparseMaxLoss, self).__init__(**kwargs)
        self.scale_factor = scale_factor
        self.label_smoothing = label_smoothing
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-10
        
        # Apply padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply efficient sparsemax
        y_pred = efficient_sparsemax(y_pred_logits)
        
        # Track accuracy
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        valid_count = tf.reduce_sum(y_mask) + epsilon
        accuracy_val = tf.reduce_sum(accuracy * y_mask) / valid_count
        self.add_metric(accuracy_val, name='accuracy', aggregation='mean')
        
        # Enhanced loss computation
        vocab_size = tf.shape(y_pred_logits)[-1]
        vocab_size_float = tf.cast(vocab_size, tf.float32)
        
        # Convert to one-hot with label smoothing
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true_smooth = (1.0 - self.label_smoothing) * y_true_oh + self.label_smoothing / vocab_size_float
        else:
            y_true_smooth = y_true_oh
        
        # Calculate loss components
        # Log-likelihood term: -<y_true, logits>
        log_term = -tf.reduce_sum(y_true_smooth * y_pred_logits, axis=-1)
        
        # Sparsemax term: 0.5 * ||y_pred||^2
        sparsemax_term = 0.5 * tf.reduce_sum(tf.square(y_pred), axis=-1)
        
        # Constant term
        constant_term = 0.5
        
        # Full loss
        point_loss = log_term + sparsemax_term + constant_term
        
        # Apply mask
        masked_loss = point_loss * y_mask
        
        # Scale the loss to improve gradient flow
        scaled_loss = masked_loss * self.scale_factor
        
        # Calculate mean loss
        mean_loss = tf.reduce_sum(scaled_loss) / valid_count
        
        # Safety check for NaN/Inf
        safe_loss = tf.where(
            tf.math.is_finite(mean_loss),
            mean_loss,
            tf.constant(1.0, dtype=tf.float32)
        )
        
        return safe_loss


class WarmupLearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler with warmup and cosine decay."""
    
    def __init__(self, warmup_steps=2000, initial_lr=2e-5, min_lr=1e-7, total_steps=100000):
        super(WarmupLearningRateScheduler, self).__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.step_counter = 0
    
    def on_batch_begin(self, batch, logs=None):
        self.step_counter += 1
        
        if self.step_counter < self.warmup_steps:
            # Linear warmup
            lr = self.min_lr + (self.initial_lr - self.min_lr) * (self.step_counter / self.warmup_steps)
        else:
            # Cosine decay after warmup
            decay_steps = self.total_steps - self.warmup_steps
            decay_step = min(self.step_counter - self.warmup_steps, decay_steps)
            cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_step / decay_steps))
            lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_decay
        
        K.set_value(self.model.optimizer.lr, lr)
    
    def on_epoch_end(self, epoch, logs=None):
        # Print current learning rate
        lr = K.get_value(self.model.optimizer.lr)
        print(f"\nCurrent learning rate: {lr:.2e}")


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
    
    # Use enhanced SparseMaxLoss with scaling and label smoothing
    outputs = EnhancedSparseMaxLoss(
        scale_factor=10.0,
        label_smoothing=0.1,
        output_axis=1
    )([y_in, model.output])
    
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    
    # Configure optimizer with better parameters
    optimizer = AdamWLRG(
        learning_rate=2e-5,  # Higher learning rate
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={40000: 1},
        clipnorm=3.0,  # Increased clipnorm
        clipvalue=30.0,  # Increased clipvalue
        epsilon=1e-8
    )
    
    # Compile with mixed precision for faster training
    mixed_precision = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(mixed_precision)
    
    train_model.compile(
        optimizer=optimizer,
        experimental_run_tf_function=False
    )
    
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)


class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('bert_model.weights')
        # Print detailed status
        print(f"\nEpoch {epoch+1} completed:")
        print(f" - Loss: {logs.get('loss'):.6f}")
        print(f" - Accuracy: {logs.get('accuracy'):.6f}")


if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    evaluator = Evaluator()
    
    # Add specialized callbacks
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    # Add warmup scheduler
    warmup_scheduler = WarmupLearningRateScheduler(
        warmup_steps=2000,
        initial_lr=2e-5,
        min_lr=1e-7,
        total_steps=100000
    )
    
    # Add TensorBoard callback for better visualization
    tensorboard = keras.callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1,
        write_graph=True,
        update_freq=100
    )
    
    # Prepare data generator
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )

    # Train with improved configuration
    history = train_model.fit(
        dataset, 
        steps_per_epoch=1000, 
        epochs=epochs, 
        callbacks=[
            evaluator,
            terminate_on_nan,
            reduce_lr,
            warmup_scheduler,
            tensorboard
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
