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
batch_size = 16  # Reduced batch size for stability
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
    """Standard cross-entropy loss with safe operations."""
    
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        
        # Create mask
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        
        # Calculate standard sparse categorical crossentropy with from_logits=True
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        
        # Apply mask
        masked_loss = loss * y_mask
        
        # Compute mean loss with safe division
        mean_loss = K.sum(masked_loss) / (K.sum(y_mask) + K.epsilon())
        
        # Calculate accuracy for monitoring
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred_softmax)
        accuracy_mean = K.sum(accuracy * y_mask) / (K.sum(y_mask) + K.epsilon())
        
        # Add accuracy as a metric
        self.add_metric(accuracy_mean, name='accuracy', aggregation='mean')
        
        return mean_loss

class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Save model at each epoch
        model.save_weights('bert_model.weights')
        print(f"\nEpoch {epoch+1}: loss={logs.get('loss', 0.0):.6f}, acc={logs.get('accuracy', 0.0):.6f}")

# Initialize model outside strategy scope for safety
bert = None
model = None
train_model = None

# Try with different strategy
try:
    # Use a simple strategy first
    strategy = tf.distribute.MirroredStrategy()
    
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
        
        # Use standard cross entropy
        outputs = CrossEntropy(1)([y_in, model.output])
        train_model = keras.models.Model(model.inputs + [y_in], outputs)

        # Configure a simple optimizer
        optimizer = Adam(
            learning_rate=1e-5,
            epsilon=1e-8,
            clipnorm=1.0,
            clipvalue=10.0
        )
        
        # Compile with minimal options
        train_model.compile(optimizer=optimizer)
        
        # Load weights
        bert.load_weights_from_checkpoint(checkpoint_path)
except Exception as e:
    print(f"Strategy setup error: {e}")
    raise

if __name__ == '__main__':
    print("GPUs available:", tf.config.list_physical_devices('GPU'))
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Add terminate_on_nan
    terminate_on_nan = keras.callbacks.TerminateOnNaN()
    
    # Try running with exception handling
    try:
        # Prepare data generator with smaller batch size
        train_generator = data_generator(corpus(), batch_size, 10**5)
        dataset = train_generator.to_dataset(
            types=('float32', 'float32', 'float32'),
            shapes=([None], [None], [None]),
            names=('Input-Token', 'Input-Segment', 'Input-Label'),
            padded_batch=True
        )
        
        # Use prefetch
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
        # Train with minimal configuration
        history = train_model.fit(
            dataset, 
            steps_per_epoch=200,  # Reduced steps for faster epochs
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
        
        print("Saving to:", os.getcwd())
        print("History keys:", history.history.keys())
        print("Accuracy values:", history.history.get('accuracy', []))
        print("Loss values:", history.history.get('loss', []))
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(history.history)
        metrics_df.to_csv('/workspace/Quant/training_metrics.csv', index=False)
    
    except Exception as e:
        print(f"Training error: {e}")
        # Try to save model even if there was an error
        if model is not None:
            model.save_weights('bert_model.weights')
else:
    if model is not None:
        model.save_weights('bert_model.weights')
    metrics_df.to_csv('/workspace/Quant/training_metrics.csv', index=False)

