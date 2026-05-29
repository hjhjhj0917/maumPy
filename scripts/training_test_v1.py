# -*- coding: utf-8 -*-

disease = "depression"  # addiction, depression, anxiety

import os
import json
import random
import platform
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

# =========================================================
# 환경 설정
# =========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("실행 시각:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print("\n===== 시스템 정보 =====")
print("OS:", platform.platform())
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU 없음")

# =========================================================
# 감정 키워드
# =========================================================

emotion_keywords = [
    "우울",
    "불안",
    "무기력",
    "죽고",
    "자살",
    "외롭",
    "눈물",
    "괴롭",
    "힘들",
    "포기",
    "자해",
    "절망",
    "공허",
    "짜증",
    "스트레스",
    "불면",
    "상처",
    "분노",
    "슬프",
    "고통"
]

# =========================================================
# 텍스트 전처리
# =========================================================

def preprocess_text(paragraphs):

    important_sentences = []

    for token in paragraphs:

        speaker = token.get("paragraph_speaker", "")
        text = token.get("paragraph_text", "").strip()

        if not text:
            continue

        # 상담사 발화 제거
        if "상담사" in speaker:
            continue

        # 감정 키워드 포함 문장만 추출
        if any(keyword in text for keyword in emotion_keywords):
            important_sentences.append(text)

    # 키워드 문장이 없으면 전체 일부 사용
    if len(important_sentences) == 0:

        fallback = []

        for token in paragraphs:

            speaker = token.get("paragraph_speaker", "")
            text = token.get("paragraph_text", "").strip()

            if "상담사" in speaker:
                continue

            fallback.append(text)

        important_sentences = fallback[:20]

    return " ".join(important_sentences)

# =========================================================
# 데이터 로드
# =========================================================

folder_path = "./data/training"

json_files = [
    f for f in os.listdir(folder_path)
    if f.endswith(".json")
]

texts = []
labels = []
filenames = []

print("\n===== 데이터 로딩 시작 =====")

for json_file in json_files:

    file_path = os.path.join(folder_path, json_file)

    try:

        with open(file_path, "r", encoding="utf-8") as f:

            js = json.load(f)

            label_val = js.get(disease, 0)

            if label_val is None:
                label_val = 0

            label = min(max(int(label_val), 0), 3)

            paragraphs = js.get("paragraph", [])

            sentence = preprocess_text(paragraphs)

            # 빈 데이터 제거
            if len(sentence.strip()) < 5:
                continue

            texts.append(sentence)
            labels.append(label)
            filenames.append(json_file)

    except Exception as e:
        print(f"에러 발생: {json_file} | {e}")

print(f"\n전체 데이터 수: {len(texts)}")

# =========================================================
# 데이터프레임 생성
# =========================================================

df = pd.DataFrame({
    "filename": filenames,
    "input": texts,
    "label": labels
})

print("\n===== 클래스 분포 =====")
print(df["label"].value_counts())

# =========================================================
# Oversampling
# =========================================================

print("\n===== Oversampling 시작 =====")

target_counts = {
    0: 732,
    1: 350,
    2: 300,
    3: 250
}

balanced_dfs = []

for label in sorted(df["label"].unique()):

    class_df = df[df["label"] == label]

    current_count = len(class_df)

    target_count = target_counts[label]

    # downsampling
    if current_count > target_count:

        sampled_df = resample(
            class_df,
            replace=False,
            n_samples=target_count,
            random_state=SEED
        )

    # oversampling
    elif current_count < target_count:

        sampled_df = resample(
            class_df,
            replace=True,
            n_samples=target_count,
            random_state=SEED
        )

    else:
        sampled_df = class_df

    balanced_dfs.append(sampled_df)

df_balanced = pd.concat(balanced_dfs)

df_balanced = df_balanced.sample(
    frac=1,
    random_state=SEED
).reset_index(drop=True)

print("\n===== Oversampling 이후 분포 =====")
print(df_balanced["label"].value_counts())

# =========================================================
# Train / Test Split
# =========================================================

train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    random_state=SEED,
    stratify=df_balanced["label"]
)

print("\n===== Train 분포 =====")
print(train_df["label"].value_counts())

print("\n===== Test 분포 =====")
print(test_df["label"].value_counts())

# =========================================================
# Class Weight 계산
# =========================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["label"]),
    y=train_df["label"]
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

print("\n===== Class Weights =====")
print(class_weights)

# =========================================================
# HuggingFace Dataset
# =========================================================

train_dataset = Dataset.from_pandas(
    train_df.reset_index(drop=True)
)

test_dataset = Dataset.from_pandas(
    test_df.reset_index(drop=True)
)

# =========================================================
# 모델
# =========================================================

model_name = "klue/roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4
)

# =========================================================
# Tokenize
# =========================================================

def tokenize_function(examples):

    return tokenizer(
        examples["input"],
        truncation=True,
        padding=False,
        max_length=256
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

remove_columns = [
    col for col in [
        "input",
        "filename",
        "__index_level_0__"
    ]
    if col in train_dataset.column_names
]

train_dataset = train_dataset.remove_columns(remove_columns)
test_dataset = test_dataset.remove_columns(remove_columns)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# =========================================================
# Metrics
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, preds)

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro"
    )

    weighted_f1 = f1_score(
        labels,
        preds,
        average="weighted"
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

# =========================================================
# Weighted Loss Trainer
# =========================================================

class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs
    ):

        labels = inputs.get("labels")

        outputs = model(**inputs)

        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# =========================================================
# TrainingArguments
# =========================================================

training_args = TrainingArguments(
    output_dir="./results",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=10,

    weight_decay=0.01,

    logging_steps=20,

    fp16=torch.cuda.is_available(),

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",
    greater_is_better=True,

    save_total_limit=2,

    report_to=[]
)

# =========================================================
# Trainer
# =========================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = WeightedTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ]
)

# =========================================================
# 학습 시작
# =========================================================

print("\n===== 모델 학습 시작 =====")

trainer.train()

# =========================================================
# 저장
# =========================================================

save_path = f"./trained_model_{disease}"

trainer.save_model(save_path)

tokenizer.save_pretrained(save_path)

print("\n모델 저장 완료")

# =========================================================
# 평가
# =========================================================

print("\n===== 최종 평가 =====")

predictions = trainer.predict(test_dataset)

preds = np.argmax(
    predictions.predictions,
    axis=-1
)

labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)

macro_f1 = f1_score(
    labels,
    preds,
    average="macro"
)

weighted_f1 = f1_score(
    labels,
    preds,
    average="weighted"
)

print("\n==================================")
print("4-Class Classification Result")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
print("==================================\n")

# =========================================================
# 상세 리포트
# =========================================================

print(classification_report(
    labels,
    preds,
    digits=4
))

# =========================================================
# Confusion Matrix
# =========================================================

cm = confusion_matrix(labels, preds)

print("\n===== Confusion Matrix =====")
print(cm)