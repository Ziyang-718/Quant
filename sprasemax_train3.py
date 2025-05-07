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
    """Improved sparsemax implementation with numerical stability.
    
    Args:
        logits: Input tensor
        axis: Axis along which to apply sparsemax
        
    Returns:
        Sparsemax tensor with same shape as logits
    """
    # Add small epsilon for numerical stability
    eps = 1e-10
    
    # Cast to float32 for consistent precision
    logits = tf.cast(logits, tf.float32)
    
    # 1) Stabilize by subtracting max
    max_logits = tf.reduce_max(logits, axis=axis, keepdims=True)
    logits_stabilized = logits - max_logits
    
    # 2) Sort values in descending order
    z_sorted = tf.sort(logits_stabilized, axis=axis, direction='DESCENDING')
    
    # 3) Compute cumulative sum
    z_cumsum = tf.cumsum(z_sorted, axis=axis)
    
    # Get dimension size
    dim = tf.shape(logits)[axis]
    dim_float = tf.cast(dim, tf.float32)
    
    # Create k values (1 to dim)
    k_indices = tf.range(1, dim_float + 1, dtype=tf.float32)
    
    # Reshape k for broadcasting with z_sorted and z_cumsum
    # Create shape [1,1,...,dim,...,1] with dim at the specified axis
    k_shape = tf.ones_like(tf.shape(logits), dtype=tf.int32)
    k_shape = tf.tensor_scatter_nd_update(k_shape, [[axis]], [dim])
    k = tf.reshape(k_indices, k_shape)
    
    # 4) Compute threshold condition: 1 + k*z > cumsum
    threshold_condition = 1.0 + k * z_sorted > z_cumsum
    
    # Ensure at least one element is selected (important for numerical stability)
    any_selected = tf.reduce_any(threshold_condition, axis=axis, keepdims=True)
    # If no elements selected, select the first one
    safe_condition = tf.logical_or(
        threshold_condition,
        tf.logical_and(tf.equal(k, 1.0), tf.logical_not(any_selected))
    )
    
    # 5) Count elements that satisfy condition
    k_z = tf.reduce_sum(tf.cast(safe_condition, tf.int32), axis=axis, keepdims=True)
    # Ensure k_z is at least 1 to avoid division by zero
    k_z_safe = tf.maximum(k_z, 1)
    k_z_float = tf.cast(k_z_safe, tf.float32)
    
    # 6) Compute threshold tau
    # Use safe approach with masks instead of gather_nd
    # Create mask that selects only the k_z-th element for each sample
    range_indices = tf.range(1, dim + 1, dtype=tf.int32)
    range_indices = tf.reshape(range_indices, k_shape)
    mask = tf.equal(range_indices, k_z)
    
    # Compute cumsum up to k_z
    masked_cumsum = tf.cast(mask, tf.float32) * z_cumsum
    # Sum all values (only the masked value will contribute)
    tau_sum = tf.reduce_sum(masked_cumsum, axis=axis, keepdims=True)
    
    # Calculate tau: (tau_sum - 1) / k_z
    tau = (tau_sum - 1.0) / k_z_float
    
    # 7) Final projection
    projection = tf.maximum(0.0, logits_stabilized - tau)
    
    # Normalize to ensure sum to 1.0 (important for numerical stability)
    projection_sum = tf.reduce_sum(projection, axis=axis, keepdims=True)
    normalized_projection = projection / (projection_sum + eps)
    
    # Replace NaN values with uniform distribution
    is_nan = tf.math.is_nan(normalized_projection)
    uniform_value = 1.0 / dim_float
    safe_projection = tf.where(
        is_nan,
        tf.ones_like(normalized_projection) * uniform_value,
        normalized_projection
    )
    
    return safe_projection

class SparseMaxLoss(Loss):
    """Numerically stable SparseMax loss implementation."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        eps = 1e-10  # Small epsilon for numerical stability
        
        # Create mask for padding tokens
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply sparsemax activation
        y_pred = sparsemax(y_pred_logits)
        
        # Calculate and track accuracy metric
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy_value = K.sum(accuracy * y_mask) / (K.sum(y_mask) + eps)
        self.add_metric(accuracy_value, name='accuracy', aggregation='mean')
        
        # Implement sparsemax loss using one-hot encoding
        # This avoids potential issues with gather operations
        y_true_int = tf.cast(y_true, tf.int32)
        vocab_size = tf.shape(y_pred_logits)[-1]
        
        # Create one-hot representation of true labels
        y_true_one_hot = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
        
        # Calculate the dot product of true labels and logits
        z_y = tf.reduce_sum(y_true_one_hot * y_pred_logits, axis=-1)
        
        # Calculate squared norm of predicted distribution
        squared_norm = tf.reduce_sum(tf.square(y_pred), axis=-1)
        
        # Calculate point-wise loss: -z_y + 0.5 * squared_norm + 0.5
        point_loss = -z_y + 0.5 * squared_norm + 0.5
        
        # Apply padding mask
        masked_loss = point_loss * y_mask
        
        # Clip values to avoid numerical instability
        clipped_loss = tf.clip_by_value(masked_loss, -1e3, 1e3)
        
        # Replace any NaN values
        safe_loss = tf.where(
            tf.math.is_finite(clipped_loss),
            clipped_loss,
            tf.zeros_like(clipped_loss)
        )
        
        # Calculate mean loss over non-padding tokens
        mean_loss = tf.reduce_sum(safe_loss) / (tf.reduce_sum(y_mask) + eps)
        
        # Final safety check
        final_loss = tf.where(
            tf.math.is_finite(mean_loss),
            mean_loss,
            tf.constant(0.1, dtype=tf.float32)  # Default small value if NaN occurs
        )
        
        return final_loss

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
    # Use the SparseMaxLoss
    outputs = SparseMaxLoss(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    
    # Configure optimizer with enhanced stability
    optimizer = AdamWLRG(
        learning_rate=5e-6,  # Reduced for stability
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={40000: 1},  # Longer schedule
        clipnorm=1.0,  # Add gradient norm clipping
        clipvalue=10.0,  # Add gradient value clipping
        epsilon=1e-8  # Increase epsilon for numerical stability
    )
    
    # Compile with experimental options for stability
    train_model.compile(
        optimizer=optimizer,
        experimental_run_tf_function=False  # Can help with complex custom losses
    )
    
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)

class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('bert_model.weights')
        # Print current status
        print(f"\nEpoch {epoch+1} completed: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}")

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    evaluator = Evaluator()
    
    # Add callbacks for stability and monitoring
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,  # Allow more epochs before stopping
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )
    
    # Add callback to stop training if NaN detected
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    # Add callback to reduce learning rate when loss plateaus
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,  # Reduce by half
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    
    # Prepare data generator
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )

    # Train the model with improved monitoring
    history = train_model.fit(
        dataset, 
        steps_per_epoch=1000, 
        epochs=epochs, 
        callbacks=[
            evaluator,
            early_stopping,
            terminate_on_nan,
            reduce_lr
        ],
        verbose=1
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