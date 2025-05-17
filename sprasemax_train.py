#! -*- coding: utf-8 -*-
# 词级别的中文 RoFormer 预训练 + MLM 任务 + SparsemaxLoss
import os
os.environ['TF_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator, text_segmentate
from tensorflow.keras.optimizers import Adam
from bert4keras.optimizers import extend_with_weight_decay, extend_with_piecewise_linear_lr, extend_with_gradient_accumulation
import tensorflow_addons as tfa
import jieba
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset

jieba.initialize()

# 设置 GPU growth
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 分布式策略
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# 超参数
maxlen = 512
batch_size = 64
epochs = 100

# 模型文件路径
config_path = 'chinese_wobert_plus_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_wobert_plus_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_wobert_plus_L-12_H-768_A-12/vocab.txt'

# 语料生成器

def corpus():
    ds = Dataset.from_file('kaggle_dataset/train/dataset.arrow')
    for item in ds:
        for piece in text_process(item['text']):
            yield piece

def text_process(text):
    segments = text_segmentate(text, 32, u'。\n')
    buf = ''
    for seg in segments:
        if buf and len(buf) + len(seg) > maxlen * 1.5:
            yield buf
            buf = ''
        buf += seg
    if buf:
        yield buf

# 分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

# 随机掩码

def random_masking(token_ids):
    rands = np.random.rand(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.randint(1, tokenizer._vocab_size))
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target

# 数据生成
class MLMDataset(DataGenerator):
    def __iter__(self, random=False):
        for is_end, text in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            src, tgt = random_masking(token_ids)
            yield np.array(src), np.array(segment_ids), np.array(tgt)

# 自定义 SparsemaxLoss
class SparsemaxLoss(Loss):
    def __init__(self, from_logits=True, **kwargs):
        super().__init__(**kwargs)
        self.from_logits = from_logits

    def call(self, y_true, y_pred_logits):
        # y_pred_logits: [batch, seq, vocab]
        # y_true:   [batch, seq]
        if self.from_logits:
            y_pred = tfa.activations.sparsemax(y_pred_logits)
        else:
            y_pred = y_pred_logits
        # 只计算非0标签位置的 loss
        mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        # one-hot y_true
        vocab_size = K.shape(y_pred)[-1]
        y_true_oh = K.one_hot(K.cast(y_true, 'int32'), vocab_size)
        # sparsemax loss: ||p||^2 / 2 - z_y + constant, here simplified
        squared_norm = K.sum(K.square(y_pred), axis=-1)
        z_y = K.sum(y_pred * y_true_oh, axis=-1)
        per_token = 0.5 * squared_norm - z_y
        loss = per_token * mask
        return K.sum(loss) / (K.sum(mask) + 1e-6)

with strategy.scope():
    # 构建 RoFormer 模型
    bert = build_transformer_model(
        config_path,
        checkpoint_path=None,
        model='roformer',
        with_mlm='linear',
        return_keras_model=False,
        ignore_invalid_weights=True
    )
    base_model = bert.model  # takes [token_ids, segment_ids], outputs logits

    # 标签输入
    y_in = keras.layers.Input(shape=(None,), dtype='int32', name='Input-Label')

    # 输出 logits
    logits = base_model.output  # [batch, seq_len, vocab_size]

    # 定义训练模型
    train_model = keras.models.Model(
        inputs = base_model.inputs + [y_in],
        outputs= logits
    )

    # 优化器：AdamW + 线性分段 LR + weight decay + grad accumulation
    AdamW = extend_with_weight_decay(Adam)
    PAdamW = extend_with_piecewise_linear_lr(AdamW)
    AdamWLRG = extend_with_gradient_accumulation(PAdamW)

    optimizer = AdamWLRG(
        learning_rate=1e-5,
        weight_decay_rate=0.01,
        exclude_from_weight_decay=['Norm', 'bias'],
        grad_accum_steps=4,
        lr_schedule={20000: 1}
    )

    # 编译模型
    train_model.compile(
        optimizer=optimizer,
        loss=SparsemaxLoss(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    )

    # 加载预训练权重
    bert.load_weights_from_checkpoint(checkpoint_path)

# Callback 保存权重
class Evaluator(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_model.save_weights('bert_model.weights')

if __name__ == '__main__':
    evaluator = Evaluator()
    # 构建 Dataset
    train_gen = MLMDataset(corpus(), batch_size)
    dataset = train_gen.to_dataset(
        types=('int32','int32','int32'),
        shapes=([None],[None],[None]),
        names=('Input-Token','Input-Segment','Input-Label'),
        padded_batch=True
    )

    # 训练
    history = train_model.fit(
        dataset,
        steps_per_epoch=1000,
        epochs=epochs,
        callbacks=[evaluator]
    )

    # 结果可视化
    plt.figure(figsize=(12,5))
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['loss'], label='Loss')
    plt.title('Training Metrics')
    plt.xlabel('Epoch'); plt.ylabel('Value'); plt.legend()
    plt.savefig('sparsemax_metrics.png')
    plt.show()

    # 保存历史
    pd.DataFrame(history.history).to_csv('training_metrics.csv', index=False)
else:
    train_model.load_weights('bert_model.weights')
