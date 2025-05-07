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
    """Improved sparsemax implementation that handles distributed training better.
    
    This implementation correctly handles tensor shapes during distributed training.
    """
    # 1) Stabilize the input
    logits = tf.cast(logits, tf.float32)
    logits_stabilized = logits - tf.reduce_max(logits, axis=axis, keepdims=True)

    # 2) Sort values
    z_sorted = tf.sort(logits_stabilized, axis=axis, direction='DESCENDING')

    # 3) Cumulative sum with careful dimension handling
    z_cumsum = tf.cumsum(z_sorted, axis=axis)
    
    # Get dimension sizes statically where possible
    input_shape = tf.shape(logits)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    vocab_size = input_shape[2]  # This should be the size along the axis=-1
    
    # Create k values [1, 2, 3, ..., vocab_size]
    k = tf.range(1, vocab_size + 1, dtype=tf.float32)
    # Make sure k has proper shape for broadcasting
    k = tf.reshape(k, [1, 1, vocab_size])  # [1, 1, vocab_size] for broadcasting
    
    # 4) Threshold condition (avoid broadcasting issues)
    z_check = 1.0 + k * z_sorted > z_cumsum
    
    # Calculate k_z (number of non-zero elements) in a way that preserves batch dimensions
    k_z = tf.reduce_sum(tf.cast(z_check, tf.int32), axis=axis, keepdims=True)  # [batch, seq_len, 1]
    
    # 5) Compute threshold tau
    # Critical: Make sure to properly handle the shapes here to avoid the reshape error
    # Prepare for gather operation
    # Create indices for each element in the batch
    batch_indices = tf.range(batch_size)  # [batch_size]
    seq_indices = tf.range(seq_len)  # [seq_len]
    
    # Create a mesh grid of indices
    mesh_batch, mesh_seq = tf.meshgrid(batch_indices, seq_indices, indexing='ij')  # [batch_size, seq_len]
    
    # Flatten mesh grid and k_z
    flat_batch_indices = tf.reshape(mesh_batch, [-1])  # [batch_size * seq_len]
    flat_seq_indices = tf.reshape(mesh_seq, [-1])  # [batch_size * seq_len]
    flat_k_z = tf.reshape(k_z - 1, [-1])  # [batch_size * seq_len]
    
    # Stack indices for gather_nd
    indices = tf.stack([flat_batch_indices, flat_seq_indices, flat_k_z], axis=1)  # [batch_size * seq_len, 3]
    
    # Gather the cumulative sum values at the threshold
    # Reshape z_cumsum to match the indices
    z_cumsum_3d = tf.reshape(z_cumsum, [batch_size, seq_len, vocab_size])
    tau_sum = tf.gather_nd(z_cumsum_3d, indices)  # [batch_size * seq_len]
    
    # Reshape to match original batch and sequence dimensions
    tau_sum = tf.reshape(tau_sum, [batch_size, seq_len])  # [batch_size, seq_len]
    
    # Get the corresponding k_z values for division
    k_z_2d = tf.reshape(k_z, [batch_size, seq_len])  # [batch_size, seq_len]
    
    # Calculate the threshold
    tau = (tau_sum - 1.0) / tf.cast(k_z_2d, tf.float32)  # [batch_size, seq_len]
    
    # Reshape tau for broadcasting against logits
    tau = tf.reshape(tau, [batch_size, seq_len, 1])  # [batch_size, seq_len, 1]
    
    # 6) Final projection
    return tf.maximum(0.0, logits_stabilized - tau)

class SparseMaxLoss(Loss):
    """Loss function for Sparsemax activation with proper shape handling for distributed training."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred_logits = inputs
        
        # Handle padding mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Apply sparsemax to logits
        y_pred = sparsemax(y_pred_logits)
        
        # Compute and track accuracy
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / (K.sum(y_mask) + K.epsilon())
        self.add_metric(accuracy, name='accuracy', aggregation='mean')
        
        # Get shape information
        batch_size = tf.shape(y_true)[0]
        seq_len = tf.shape(y_true)[1]
        vocab_size = tf.shape(y_pred_logits)[2]
        
        # Reshape tensors carefully
        y_true_flat = tf.reshape(y_true, [-1])  # [batch_size * seq_len]
        y_mask_flat = tf.reshape(y_mask, [-1])  # [batch_size * seq_len]
        
        # Identify valid positions (non-padding)
        valid_positions = tf.cast(y_mask_flat, tf.bool)
        
        # Filter out padded positions
        valid_y_true = tf.boolean_mask(y_true_flat, valid_positions)
        valid_y_true = tf.cast(valid_y_true, tf.int32)
        
        # Find the position indices in the flattened tensor
        position_indices = tf.range(batch_size * seq_len)
        valid_positions_indices = tf.boolean_mask(position_indices, valid_positions)
        
        # Calculate batch and position indices from flattened indices
        valid_batch_indices = valid_positions_indices // seq_len
        valid_seq_indices = valid_positions_indices % seq_len
        
        # Stack indices for gather
        gather_indices = tf.stack(
            [valid_batch_indices, valid_seq_indices, valid_y_true], 
            axis=1
        )  # [num_valid, 3]
        
        # Get logits for true classes
        z_y = tf.gather_nd(y_pred_logits, gather_indices)  # [num_valid]
        
        # Calculate squared norm term for valid positions
        valid_preds = tf.gather_nd(y_pred, tf.stack([valid_batch_indices, valid_seq_indices], axis=1))
        squared_norm = tf.reduce_sum(tf.square(valid_preds), axis=1)  # [num_valid]
        
        # Compute the loss
        point_loss = -z_y + 0.5 * squared_norm + 0.5
        
        # Return mean loss
        return tf.reduce_mean(point_loss)

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
    metrics_df.to_csv('/workspace/Quant/Sparsemax_training_metrics.csv', index=False)

else:
    model.load_weights('bert_model.weights')
