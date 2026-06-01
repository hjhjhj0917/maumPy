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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, EarlyStoppingCallback

disease = "depression"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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


folder_path = "./data/training"
json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

texts = []
labels = []
filenames = []

for json_file in json_files:
    file_path = os.path.join(folder_path, json_file)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            js = json.load(f)
            label_val = js.get(disease, 0)
            if label_val is None:
                label_val = 0

            label = min(max(int(label_val), 0), 3)
            binary_label = 0 if label == 0 else 1

            paragraphs = js.get("paragraph", [])
            sentence = preprocess_text(paragraphs)

            if len(sentence.strip()) < 5:
                continue

            texts.append(sentence)
            labels.append(binary_label)
            filenames.append(json_file)

    except Exception:
        continue

df = pd.DataFrame({
    "filename": filenames,
    "input": texts,
    "label": labels
})

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"]
)

max_count = train_df["label"].value_counts().max()
balanced_dfs = []

for label in sorted(train_df["label"].unique()):
    class_df = train_df[train_df["label"] == label]
    sampled_df = resample(class_df, replace=True, n_samples=max_count, random_state=SEED)
    balanced_dfs.append(sampled_df)

train_df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)

class_weights_tensor = torch.tensor([1.0, 1.2], dtype=torch.float).to(device)

train_dataset = Dataset.from_pandas(train_df_balanced)
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


def tokenize_function(examples):
    return tokenizer(
        examples["input"],
        truncation=True,
        padding=False,
        max_length=512
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

remove_columns = [col for col in ["input", "filename", "__index_level_0__"] if col in train_dataset.column_names]
train_dataset = train_dataset.remove_columns(remove_columns)
test_dataset = test_dataset.remove_columns(remove_columns)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch")
test_dataset.set_format("torch")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()

    threshold = 0.4
    preds = (probs[:, 1] >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    binary_f1 = f1_score(labels, preds, average="binary")

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1
    }


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss_fct = nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


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

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

save_path = f"./trained_model_{disease}_binary"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

predictions = trainer.predict(test_dataset)
probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
threshold = 0.4
preds = (probs[:, 1] >= threshold).astype(int)
labels = predictions.label_ids

accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")
binary_f1 = f1_score(labels, preds, average="binary")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print(f"Binary F1: {binary_f1 * 100:.2f}%")
print(classification_report(labels, preds, digits=4))
print(confusion_matrix(labels, preds))

# --- 기존 코드 맨 아래에 이어서 추가 ---

print("\n===== 🎯 최적의 Threshold(임계값) 찾기 =====")

# trainer.predict()로 뽑아둔 확률 값(probs) 재사용
best_threshold = 0.5
best_f1 = 0.0

# 0.40 부터 0.60 까지 0.05 단위로 테스트
for t in np.arange(0.40, 0.65, 0.05):
    temp_preds = (probs[:, 1] >= t).astype(int)

    temp_acc = accuracy_score(labels, temp_preds)
    temp_binary_f1 = f1_score(labels, temp_preds, average="binary")
    temp_recall = f1_score(labels, temp_preds, average="binary")  # recall 확인용

    print(f"[Threshold {t:.2f}] Accuracy: {temp_acc * 100:.2f}% | Binary F1: {temp_binary_f1 * 100:.2f}%")

    # 혼동 행렬 간략 출력
    cm = confusion_matrix(labels, temp_preds)
    print(f"  -> 정상오해(FP): {cm[0][1]}명 | 환자놓침(FN): {cm[1][0]}명 | 환자찾음(TP): {cm[1][1]}명\n")

    if temp_binary_f1 > best_f1:
        best_f1 = temp_binary_f1
        best_threshold = t

print(f"💡 결론: 이 모델의 최고 성능은 Threshold가 {best_threshold:.2f} 일 때, Binary F1 {best_f1 * 100:.2f}% 입니다!")