#! -*- coding: utf-8 -*-
# 词级别的中文RoFormer预训练
# MLM任务

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import json
import numpy as np
import tensorflow as tf
from bert4keras.backend2 import keras, K
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

# Use backend's fixed sparsemax implementation directly
from bert4keras.backend import sparsemax

class SparsemaxLoss(Loss):
    """Improved sparsemax loss with better stability for distributed training."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        epsilon = 1e-12
        
        # Create padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply backend's sparsemax function
        y_pred = sparsemax(y_pred_logits)
        
        # Calculate and track accuracy with better masking
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        valid_count = tf.reduce_sum(y_mask) + epsilon
        accuracy_val = tf.reduce_sum(accuracy * y_mask) / valid_count
        self.add_metric(accuracy_val, name='accuracy', aggregation='mean')
        
        # Convert true labels to one-hot encoding
        y_true_int = tf.cast(y_true, tf.int32)
        vocab_size = tf.shape(y_pred_logits)[-1]
        y_true_oh = tf.one_hot(y_true_int, vocab_size, dtype=tf.float32)
        
        # Add label smoothing for better training stability
        label_smoothing = 0.1
        y_true_smooth = (1.0 - label_smoothing) * y_true_oh + label_smoothing / tf.cast(vocab_size, tf.float32)
        
        # Compute standard sparsemax loss components
        # First term: -<y_true, logits>
        log_term = -tf.reduce_sum(y_true_smooth * y_pred_logits, axis=-1)
        
        # Second term: 0.5 * ||y_pred||^2
        squared_norm = 0.5 * tf.reduce_sum(tf.square(y_pred), axis=-1)
        
        # Constant term
        constant_term = 0.5
        
        # Full loss calculation
        point_loss = log_term + squared_norm + constant_term
        
        # Scale the loss to ensure proper gradient magnitude
        scale_factor = 10.0
        scaled_point_loss = point_loss * scale_factor
        
        # Apply mask to consider only non-padding tokens
        masked_loss = scaled_point_loss * y_mask
        
        # Calculate mean loss over valid positions
        mean_loss = tf.reduce_sum(masked_loss) / (tf.reduce_sum(y_mask) + epsilon)
        
        # Final safety check for NaN/Inf values
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
    def __init__(self, save_interval=1):
        super(Evaluator, self).__init__()
        self.save_interval = save_interval
    
    def on_epoch_end(self, epoch, logs=None):
        # Save model weights at specified intervals to reduce I/O overhead
        if (epoch + 1) % self.save_interval == 0:
            model.save_weights('bert_model.weights')
        
        # Print minimal status information to reduce callback overhead
        print(f"\nEpoch {epoch+1}: loss={logs.get('loss'):.6f}, acc={logs.get('accuracy'):.6f}")

# Learning rate warmup callback
class WarmUpLearningRate(keras.callbacks.Callback):
    def __init__(self, warmup_steps=2000, initial_lr=1e-5, min_lr=1e-7):
        super(WarmUpLearningRate, self).__init__()
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.step_counter = 0
    
    def on_batch_begin(self, batch, logs=None):
        # Skip the warmup if we're resuming training
        if hasattr(self.model, 'optimizer') and hasattr(self.model.optimizer, 'lr'):
            self.step_counter += 1
            
            if self.step_counter < self.warmup_steps:
                # Linear warmup
                lr = self.min_lr + (self.initial_lr - self.min_lr) * (self.step_counter / self.warmup_steps)
                K.set_value(self.model.optimizer.lr, lr)

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
    
    # Use the improved SparsemaxLoss
    outputs = SparsemaxLoss(1)([y_in, model.output])
    train_model = keras.models.Model(model.inputs + [y_in], outputs)

    AdamW = extend_with_weight_decay(Adam, name='AdamW')
    AdamWLR = extend_with_piecewise_linear_lr(AdamW, name='AdamWLR')
    AdamWLRG = extend_with_gradient_accumulation(AdamWLR, name='AdamWLRG')
    
    # Configure optimizer with more effective parameters
    optimizer = AdamWLRG(
        learning_rate=1e-5,  # Start with a lower learning rate for warmup
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={40000: 0.5, 80000: 0.1},  # Longer schedule
        epsilon=1e-8,  # Better numerical stability
        clipnorm=3.0,  # Increased gradient clipping for better stability
        clipvalue=30.0  # Increased value clipping
    )
    
    # Compile with minimal options for better efficiency
    train_model.compile(optimizer=optimizer)
    
    train_model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    
    # Initialize callbacks with improved efficiency
    evaluator = Evaluator(save_interval=5)  # Save weights less frequently
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    warmup_lr = WarmUpLearningRate(warmup_steps=2000, initial_lr=1e-4, min_lr=1e-7)
    
    # Prepare data generator with specific batch size
    train_generator = data_generator(corpus(), batch_size, 10**5)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32', 'float32'),
        shapes=([None], [None], [None]),
        names=('Input-Token', 'Input-Segment', 'Input-Label'),
        padded_batch=True
    )
    
    # Use performance-optimized dataset options
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Train with optimized callbacks
    history = train_model.fit(
        dataset, 
        steps_per_epoch=1000, 
        epochs=epochs, 
        callbacks=[
            evaluator,
            terminate_on_nan,
            warmup_lr
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
