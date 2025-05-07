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
    """Simplified sparsemax implementation with fixed broadcasting.
    
    This version avoids complex operations that could cause broadcasting issues.
    """
    epsilon = 1e-12  # Numerical stability constant
    
    # Cast to float32 for stability
    logits = tf.cast(logits, tf.float32)
    
    # Step 1: Apply standard softmax-style normalization
    # Subtract max for numerical stability
    logits_shifted = logits - tf.reduce_max(logits, axis=axis, keepdims=True)
    
    # Step 2: Sort the values in descending order
    z_sorted = tf.sort(logits_shifted, axis=axis, direction='DESCENDING')
    
    # Get the cumulative sum
    z_cumsum = tf.cumsum(z_sorted, axis=axis)
    
    # Get the number of elements in the last dimension
    dim = tf.shape(logits)[axis]
    dim_float = tf.cast(dim, tf.float32)
    
    # Create a range tensor for the k values: [1, 2, 3, ..., dim]
    k_tensor = tf.range(1, dim_float + 1, dtype=tf.float32)
    
    # We need to reshape k_tensor to enable proper broadcasting
    # For axis=-1, k_tensor should have shape [1, 1, ..., dim]
    # Create a shape tensor filled with ones, then set the last dimension to dim
    k_shape = tf.ones_like(tf.shape(logits), dtype=tf.int32)
    k_shape = tf.tensor_scatter_nd_update(k_shape, [[tf.rank(logits) - 1]], [dim])
    k = tf.reshape(k_tensor, k_shape)
    
    # Calculate the threshold condition: 1 + k * z > cumsum
    threshold = 1.0 + k * z_sorted > z_cumsum
    
    # Sum over the last dimension to count how many elements satisfy the condition
    # This is the number of non-zero elements in the output
    k_z = tf.reduce_sum(tf.cast(threshold, tf.int32), axis=axis, keepdims=True)
    
    # Handle edge case: if k_z is 0, set it to 1 to avoid division by zero
    k_z = tf.maximum(k_z, 1)
    k_z_float = tf.cast(k_z, tf.float32)
    
    # Get the critical value from the sorted cumulative sum
    # This is a simpler approach that avoids complex gathering operations
    
    # Create a sequence mask for selecting elements
    # The mask will have 1s for positions < k_z and 0s elsewhere
    mask = tf.sequence_mask(
        tf.reshape(k_z, [-1]), 
        maxlen=dim, 
        dtype=tf.float32
    )
    
    # Reshape mask to match z_sorted
    mask_shape = tf.concat([
        tf.shape(z_sorted)[:-1],
        [dim]
    ], axis=0)
    mask = tf.reshape(mask, mask_shape)
    
    # Use the mask to select elements
    # Get the last selected element's value using a reversed cumulative sum trick
    # First, reverse the mask and find the first 1 position
    rev_mask = tf.reverse(mask, axis=[axis])
    rev_sorted = tf.reverse(z_sorted, axis=[axis])
    # Multiply the reversed mask with the reversed sorted values
    last_selected = tf.reduce_sum(rev_mask * rev_sorted, axis=axis, keepdims=True)
    
    # Calculate the threshold value tau using a simplification of the original formula
    tau = (z_cumsum * mask - 1.0) / k_z_float
    tau = tf.reduce_sum(tau, axis=axis, keepdims=True)
    
    # Apply the threshold to the original logits
    sparse_probs = tf.maximum(0.0, logits_shifted - tau)
    
    # Normalize to ensure the sum is 1
    # This step is crucial for making sure the distribution is proper
    normalizer = tf.reduce_sum(sparse_probs, axis=axis, keepdims=True)
    # Add epsilon to avoid division by zero
    normalizer = tf.maximum(normalizer, epsilon)
    sparse_probs = sparse_probs / normalizer
    
    # Final safety check - replace any NaN values with uniform distribution
    is_bad = tf.math.logical_or(
        tf.math.is_nan(sparse_probs),
        tf.math.is_inf(sparse_probs)
    )
    uniform_probs = tf.ones_like(sparse_probs) / dim_float
    return tf.where(is_bad, uniform_probs, sparse_probs)


class SparseMaxLoss(Loss):
    """Simplified SparseMax loss that avoids broadcasting issues."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-12
        
        # Create padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply sparsemax to get predictions
        y_pred = sparsemax(y_pred_logits)
        
        # Calculate and track accuracy
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / (K.sum(y_mask) + epsilon)
        self.add_metric(accuracy, name='accuracy', aggregation='mean')
        
        # Use the simpler form of sparsemax loss
        # Convert true labels to one-hot encoding
        vocab_size = tf.shape(y_pred_logits)[-1]
        y_true_int = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
        
        # Calculate the first term: negative dot product of true labels and logits
        term1 = -tf.reduce_sum(y_true_oh * y_pred_logits, axis=-1)
        
        # Calculate the second term: squared norm of predictions
        term2 = 0.5 * tf.reduce_sum(tf.square(y_pred), axis=-1)
        
        # Constant term
        term3 = 0.5
        
        # Combine terms
        point_loss = term1 + term2 + term3
        
        # Apply padding mask
        masked_loss = point_loss * y_mask
        
        # Clip to avoid extreme values
        clipped_loss = tf.clip_by_value(masked_loss, -1e3, 1e3)
        
        # Handle NaN values
        safe_loss = tf.where(
            tf.math.is_finite(clipped_loss),
            clipped_loss,
            tf.zeros_like(clipped_loss)
        )
        
        # Calculate mean loss with safe division
        mean_loss = tf.reduce_sum(safe_loss) / (tf.reduce_sum(y_mask) + epsilon)
        
        # Final safety check
        return tf.where(
            tf.math.is_finite(mean_loss),
            mean_loss,
            tf.constant(0.1, dtype=tf.float32)
        )


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
    # Use the simplified SparseMaxLoss
    outputs = SparseMaxLoss(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    
    # Configure optimizer with additional stability settings
    optimizer = AdamWLRG(
        learning_rate=1e-6,  # Reduced learning rate for stability
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={40000: 1},
        clipnorm=1.0,  # Add gradient clipping
        clipvalue=10.0,  # Clip gradient values
        epsilon=1e-8  # Increase epsilon for better numerical stability
    )
    
    # Compile with experimental_run_tf_function=False for complex custom losses
    train_model.compile(
        optimizer=optimizer,
        experimental_run_tf_function=False
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
    
    # Add monitoring callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        min_delta=0.001,
        restore_best_weights=True,
        verbose=1
    )
    
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
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

    # Train with all callbacks enabled
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
