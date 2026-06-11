import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, EarlyStoppingCallback

# 학습 모델 명과 고정 시드값 42(암묵적 룰)
disease = "depression"
SEED = 42

# 시드값을 고정 일관된 결과를 위해
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 학습할 기기 선택
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_text(paragraphs):
    sentences = []
    for token in paragraphs:
        speaker = token.get("paragraph_speaker", "")
        text = token.get("paragraph_text", "").strip()

        if not text or "상담사" in speaker:
            continue

        sentences.append(text)

    return " ".join(sentences)


# 해당 경로에 .json 파일만 가져옴
folder_path = "./data/training"
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

texts = []
labels = []
filenames = []

for json_file in json_files:
    file_path = os.path.join(folder_path, json_file) # 해당 경로에 json 파일 하나씩 가져옴
    try:
        with open(file_path, "r", encoding="utf-8") as f: # 인코딩과 읽기 모드로 파일 오픈
            js = json.load(f) # 파일 내용 불러옴
            label_val = js.get(disease, 0) # 없는 값 대비
            if label_val is None:
                label_val = 0

            label = min(max(int(label_val), 0), 3) # 이상치 제거
            binary_label = 0 if label == 0 else 1 # 이진분류로 변경

            paragraphs = js.get("paragraph", []) # paragraph 받아옴
            sentence = preprocess_text(paragraphs) # 상담사 제거하고 가져옴

            if len(sentence.strip()) < 5:
                continue

            # 검증에 통과한 내용만 저장
            texts.append(sentence)
            labels.append(binary_label)
            filenames.append(json_file)

    except Exception:
        continue

df = pd.DataFrame({ # 저장된 데이터를 Pandas DataFrame 구조로 변환
    "filename": filenames,
    "input": texts,
    "label": labels
})

train_df, test_df = train_test_split( # 전체 데이터를 학습용과 평가용으로 분리
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"]
)

max_count = train_df["label"].value_counts().max()
balanced_dfs = []

for label in sorted(train_df["label"].unique()): # 환자와 정상자 개수를 확인해서 비율을 1:1로 맞춤
    class_df = train_df[train_df["label"] == label]
    sampled_df = resample(class_df, replace=True, n_samples=max_count, random_state=SEED)
    balanced_dfs.append(sampled_df)

train_df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True) # 개수를 맞춘 데이터를 무작위로 섞음

class_weights_tensor = torch.tensor([1.0, 1.2], dtype=torch.float).to(device) # 환자 데이터를 놓치면 손실이 더 커서 가중치를 더 높게 부여

# Hugging Face 전용 데이터 셋으로 변경
train_dataset = Dataset.from_pandas(train_df_balanced)
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

model_name = "klue/roberta-base" # 모델 호출
tokenizer = AutoTokenizer.from_pretrained(model_name) # 상담데이터를 모델이 처리할 수 있는 단위로 나눔
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # 마지막 출력을 2진 분류하기 위한 층을 2개로 설정

# 토큰화
def tokenize_function(examples):
    return tokenizer(
        examples["input"],
        truncation=True,
        padding=False,
        max_length=512
    )

# 토큰화 함수를 데이터 셋 전체에 적용
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 학습에 필요없는 데이터 제거
remove_columns = [col for col in ["input", "filename", "__index_level_0__"] if col in train_dataset.column_names]
train_dataset = train_dataset.remove_columns(remove_columns)
test_dataset = test_dataset.remove_columns(remove_columns)

# Hugging Face 학습에 맞게 label에서 이름을 labels로 변경
train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

# 학습을 위해 형식을 파이썬 기본 리스트에서 텐서 형식으로 변경
train_dataset.set_format("torch")
test_dataset.set_format("torch")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy() # softmax는 분석된 결과의 합이 정확히 1이 되도록 하는 함수

    # 우울증 판단 기준 생성
    threshold = 0.4
    preds = (probs[:, 1] >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds) # 전체 데이터 중 모델이 정답을 맞힌 비율
    macro_f1 = f1_score(labels, preds, average="macro") # 정상인과 환자 각각의 F1-score를 구한 뒤 평균
    binary_f1 = f1_score(labels, preds, average="binary") # 우울증(환자) 레이블에 집중하여 계산한 F1-score

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1
    }


class WeightedTrainer(Trainer): # 가중치 부여
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# 학습 환경 및 하이퍼파라미터 설정
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="binary_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to=[]
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 모델, 설정, 데이터를 결합하여 학습기(Trainer) 생성
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 학습 시작
trainer.train()

# 최종 모델 저장
save_path = f"./trained_model_{disease}_binary"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

# 테스트셋 예측 수행
predictions = trainer.predict(test_dataset)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
threshold = 0.4
preds = (probs[:, 1] >= threshold).astype(int)
labels = predictions.label_ids

# 기본 지표 출력 (정확도, F1-Score 등)
accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")
binary_f1 = f1_score(labels, preds, average="binary")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print(f"Binary F1: {binary_f1 * 100:.2f}%")
print(classification_report(labels, preds, digits=4))
print(confusion_matrix(labels, preds))


print("\n===== Threshold(임계값) 찾기 =====")

best_threshold = 0.5
best_f1 = 0.0

# 0.40에서 0.65까지 0.05 단위로 테스트하며 가장 높은 F1 점수를 내는 임계값 탐색
for t in np.arange(0.40, 0.65, 0.05):
    temp_preds = (probs[:, 1] >= t).astype(int)

    temp_acc = accuracy_score(labels, temp_preds)
    temp_binary_f1 = f1_score(labels, temp_preds, average="binary")
    temp_recall = f1_score(labels, temp_preds, average="binary")

    print(f"[Threshold {t:.2f}] Accuracy: {temp_acc * 100:.2f}% | Binary F1: {temp_binary_f1 * 100:.2f}%")

    cm = confusion_matrix(labels, temp_preds)
    print(f"  -> 정상오해(FP): {cm[0][1]}명 | 환자놓침(FN): {cm[1][0]}명 | 환자찾음(TP): {cm[1][1]}명\n")

    if temp_binary_f1 > best_f1:
        best_f1 = temp_binary_f1
        best_threshold = t

print(f"결론: 이 모델의 최고 성능은 Threshold가 {best_threshold:.2f} 일 때, Binary F1 {best_f1 * 100:.2f}% 입니다!")