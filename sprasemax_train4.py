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


def sparsemax_simple(logits, axis=-1):
    """Extremely simplified sparsemax implementation that avoids broadcasting issues."""
    epsilon = 1e-12
    
    # Convert to float32 for numerical stability
    logits = tf.cast(logits, tf.float32)
    
    # Standard softmax-style normalization for numerical stability
    logits_shifted = logits - tf.reduce_max(logits, axis=axis, keepdims=True)
    
    # Get the original shape of the input
    original_shape = tf.shape(logits)
    
    # Reshape to 2D for easier processing - collapse all but the last dimension
    vocab_size = original_shape[-1]
    logits_2d = tf.reshape(logits_shifted, [-1, vocab_size])
    
    # Sort in descending order
    sorted_logits = tf.sort(logits_2d, axis=-1, direction='DESCENDING')
    cumsum = tf.cumsum(sorted_logits, axis=-1)
    
    # Find the threshold
    range_idx = tf.range(1, vocab_size + 1, dtype=tf.float32)
    range_idx = tf.reshape(range_idx, [1, -1])  # [1, vocab_size]
    
    # Threshold condition: 1 + k*z > cumsum
    condition = 1.0 + range_idx * sorted_logits > cumsum
    
    # Count elements satisfying the condition for each row
    # This will have shape [batch_size]
    k = tf.reduce_sum(tf.cast(condition, tf.int32), axis=-1)
    
    # Handle case where k=0 (no elements selected)
    k = tf.maximum(k, 1)
    
    # Create a mask to extract the k-th element for each row
    # First convert k to indices
    k_indices = k - 1  # 0-based indexing
    batch_indices = tf.range(tf.shape(k)[0])
    
    # Create indices for gather
    gather_indices = tf.stack([batch_indices, k_indices], axis=1)
    
    # Gather the cumulative sums at the threshold positions
    # This gets the sum of all elements that should be non-zero
    cumsum_threshold = tf.gather_nd(cumsum, gather_indices)
    
    # Calculate tau for each row
    tau = (cumsum_threshold - 1.0) / tf.cast(k, tf.float32)
    tau = tf.reshape(tau, [-1, 1])  # Make it broadcastable
    
    # Apply the threshold
    result = tf.maximum(0.0, logits_2d - tau)
    
    # Normalize to ensure sum to 1
    result_sum = tf.reduce_sum(result, axis=-1, keepdims=True)
    result_sum = tf.maximum(result_sum, epsilon)
    result = result / result_sum
    
    # Handle any NaN values
    result = tf.where(
        tf.math.is_finite(result),
        result,
        tf.ones_like(result) / tf.cast(vocab_size, tf.float32)
    )
    
    # Reshape back to original shape
    return tf.reshape(result, original_shape)


class SparseMaxLoss(Loss):
    """Very simplified SparseMax loss implementation to avoid broadcasting issues."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-12
        
        # Create padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply simplified sparsemax
        y_pred = sparsemax_simple(y_pred_logits)
        
        # Track accuracy
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy_val = K.sum(accuracy * y_mask) / (K.sum(y_mask) + epsilon)
        self.add_metric(accuracy_val, name='accuracy', aggregation='mean')
        
        # Create one-hot encoding of true labels
        y_true_int = tf.cast(y_true, tf.int32)
        vocab_size = tf.shape(y_pred_logits)[-1]
        y_true_oh = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
        
        # Calculate dot product term (first term of the loss)
        term1 = -tf.reduce_sum(y_true_oh * y_pred_logits, axis=-1)
        
        # Calculate squared norm term (second term)
        term2 = 0.5 * tf.reduce_sum(tf.square(y_pred), axis=-1)
        
        # Add constant term
        term3 = 0.5
        
        # Calculate point-wise loss
        point_loss = term1 + term2 + term3
        
        # Apply padding mask
        masked_loss = point_loss * y_mask
        
        # Clip to avoid extreme values
        clipped_loss = tf.clip_by_value(masked_loss, -1e3, 1e3)
        
        # Replace NaN/Inf with zeros
        safe_loss = tf.where(
            tf.math.is_finite(clipped_loss),
            clipped_loss,
            tf.zeros_like(clipped_loss)
        )
        
        # Calculate mean over non-padding positions
        total_mask = tf.reduce_sum(y_mask) + epsilon
        mean_loss = tf.reduce_sum(safe_loss) / total_mask
        
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
    # Use extremely simplified SparseMaxLoss
    outputs = SparseMaxLoss(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    
    # Configure optimizer with stability settings
    optimizer = AdamWLRG(
        learning_rate=1e-6,  # Very small learning rate for stability
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={40000: 1},
        clipnorm=1.0,
        clipvalue=10.0,
        epsilon=1e-8
    )
    
    # Compile with experimental_run_tf_function=False
    train_model.compile(
        optimizer=optimizer,
        experimental_run_tf_function=False
    )
    
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)

class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('bert_model.weights')
        # Print status after each epoch
        print(f"\nEpoch {epoch+1} completed: loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}")

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    evaluator = Evaluator()
    
    # Only keep essential callbacks, removed early stopping
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=3,
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

    # Train with reduced callbacks
    history = train_model.fit(
        dataset, 
        steps_per_epoch=1000, 
        epochs=epochs, 
        callbacks=[
            evaluator,
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
