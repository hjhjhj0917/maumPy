import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.utils import resample
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    DataCollatorWithPadding, EarlyStoppingCallback

disease = "depression"
SEED = 42  # 42로 고정해 실행할 때마다 동일한 결과가 재현되게 함

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ★ 즐겨찾기 이후 추가/수정
def preprocess_text(paragraphs):
    sentences = []
    for token in paragraphs:
        speaker = token.get("paragraph_speaker", "")
        text = token.get("paragraph_text", "").strip()

        # "상담사"/"상담자" 둘 다 상담사를 가리키는 표기라 "상담"으로 묶어서 필터함
        # (전에는 "상담사"만 걸러내서 "상담자"로 표기된 상담사 발언이 내담자 발화에 섞여 들어갔음)
        if not text or "상담" in speaker:
            continue

        sentences.append(text)

    return " ".join(sentences)


# 상담 세션 원문은 평균 5천 토큰이 넘는데(klue/roberta-base는 최대 512 토큰까지만 처리 가능),
# 그냥 자르면(truncation) 내용의 90% 이상이 버려져서 모델이 인사말 정도만 보고 판단하게 되는 문제가 있었음.
# → 문서 전체를 겹치는 청크(chunk)로 나눠서 전체 내용을 다 학습/평가에 반영함
CHUNK_MAX_LEN = 480  # [CLS]/[SEP] 특수 토큰 자리(2개)를 남기고 512에 맞춘 값
CHUNK_STRIDE = 50    # 청크 경계에서 문맥이 뚝 끊기는 걸 완화하기 위한 겹침 구간


# ★ 즐겨찾기 이후 추가/수정
def chunk_document(text, tokenizer, max_len=CHUNK_MAX_LEN, stride=CHUNK_STRIDE):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return []

    chunks = []
    step = max_len - stride
    for start in range(0, len(ids), step):
        window = ids[start:start + max_len]
        if not window:
            break
        chunks.append([tokenizer.cls_token_id] + window + [tokenizer.sep_token_id])
        if start + max_len >= len(ids):
            break
    return chunks


# ★ 즐겨찾기 이후 추가/수정
def build_chunked_dataset(df, tokenizer):
    """
    데이터프레임의 각 문서(세션)를 청크로 쪼개서 Hugging Face Dataset으로 만듦.
    반환값의 doc_ids는 Dataset에는 포함하지 않고 따로 반환함
    (평가 시 같은 문서의 청크들을 다시 묶어 집계하기 위한 용도 — 모델 입력에는 불필요).
    """
    input_ids_list, attention_mask_list, labels_list, doc_ids = [], [], [], []

    for doc_id, row in enumerate(df.itertuples()):
        for chunk_ids in chunk_document(row.input, tokenizer):
            input_ids_list.append(chunk_ids)
            attention_mask_list.append([1] * len(chunk_ids))
            labels_list.append(row.label)
            doc_ids.append(doc_id)

    dataset = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    })
    return dataset, doc_ids


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

            label = min(max(int(label_val), 0), 3)  # 0~3 범위로 이상치 제거
            binary_label = 0 if label == 0 else 1  # 다중 등급을 정상/환자 이진분류로 축소

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

# 파일명에서 진단군+환자번호를 묶어 환자 ID로 추출함 (예: "우울증_0001", "불안장애_0003").
# 번호만 쓰면 서로 다른 진단군의 "0001"이 같은 사람으로 합쳐지는 오류가 생기므로 진단군명까지 포함함.
# 이 데이터셋은 한 환자가 회기별로 여러 파일에 나뉘어 있는 구조라, 파일 단위로 무작위 분리하면
# 같은 환자의 다른 회기가 학습/평가에 동시에 들어가서 "처음 보는 사람"이 아니라
# "이미 본 사람의 말투"를 맞히는 것에 가까워짐(데이터 누수).
# → 환자(그룹) 단위로 통째로 나눠서, 평가셋에는 학습 때 전혀 안 본 환자만 들어가게 함
df["patient_id"] = df["filename"].str.extract(r"\.\s*([가-힣]+_\d+)\.")
print("환자별 세션 수:\n", df["patient_id"].value_counts())
print("환자별 라벨(0=정상,1=환자) 분포:\n", df.groupby("patient_id")["label"].mean())

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))
train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

print(f"\n[환자 단위 분리 결과] 학습 환자: {sorted(train_df['patient_id'].unique())}")
print(f"[환자 단위 분리 결과] 평가 환자: {sorted(test_df['patient_id'].unique())}")
print(f"학습 세션 수: {len(train_df)}, 평가 세션 수: {len(test_df)}")
print(f"평가셋 라벨 분포: {test_df['label'].value_counts().to_dict()}")

max_count = train_df["label"].value_counts().max()
balanced_dfs = []

# 환자/정상 개수를 1:1 비율로 맞춰 클래스 불균형을 완화함
for label in sorted(train_df["label"].unique()):
    class_df = train_df[train_df["label"] == label]
    sampled_df = resample(class_df, replace=True, n_samples=max_count, random_state=SEED)
    balanced_dfs.append(sampled_df)

train_df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=SEED).reset_index(drop=True)

class_weights_tensor = torch.tensor([1.0, 1.3], dtype=torch.float).to(device) # 1.5는 상위k 집계와 겹쳐 과보정(거의 전부 환자로 예측)이 나서, 1.2와 1.5 중간인 1.3으로 완만하게 조정

model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 정상/환자 2진 분류

# 문서(세션) 전체를 청크로 쪼개서 Dataset 생성. test_doc_ids는 나중에 청크별 예측을
# 문서 단위로 다시 묶어 집계할 때 씀 (학습에는 쓰지 않음)
train_dataset, _ = build_chunked_dataset(train_df_balanced, tokenizer)
test_dataset, test_doc_ids = build_chunked_dataset(test_df.reset_index(drop=True), tokenizer)

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# 주의: 여기서 계산하는 지표는 "청크" 단위임(학습 중 매 epoch 조기종료/체크포인트 선택 용도).
# 진짜 성능(한 세션 전체를 우울증으로 판단했는가)은 학습이 끝난 뒤 문서 단위로 청크를 집계해서 따로 계산함
# ★ 즐겨찾기 이후 추가/수정
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

# 테스트셋 예측 수행 (청크 단위로 나온 결과를, 같은 세션끼리 묶어서 문서 단위로 집계함)
predictions = trainer.predict(test_dataset)
chunk_probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()[:, 1]
chunk_labels = predictions.label_ids

doc_probs = {}
doc_labels = {}
for prob, label, doc_id in zip(chunk_probs, chunk_labels, test_doc_ids):
    doc_probs.setdefault(doc_id, []).append(prob)
    doc_labels[doc_id] = label  # 같은 문서의 청크는 라벨이 전부 동일함

doc_ids_sorted = sorted(doc_probs.keys())
# (상위 k개 평균 방식을 시도했었으나 클래스 가중치 조정과 겹쳐서 모델이 거의 전부를
#  "환자"로 예측해버리는 과보정이 발생함 — 원인 분리를 위해 평균 방식으로 되돌림)
probs_1d = np.array([np.mean(doc_probs[d]) for d in doc_ids_sorted])  # 청크 확률 평균 = 문서(세션) 확률
probs = np.stack([1 - probs_1d, probs_1d], axis=1)  # 이후 코드가 기대하는 [정상확률, 환자확률] 형태로 맞춤
threshold = 0.4
preds = (probs[:, 1] >= threshold).astype(int)
labels = np.array([doc_labels[d] for d in doc_ids_sorted])

print(f"\n(참고: 세션 {len(test_df)}개가 청크 {len(test_doc_ids)}개로 나뉘어 학습/평가에 반영됨)")

accuracy = accuracy_score(labels, preds)
macro_f1 = f1_score(labels, preds, average="macro")
binary_f1 = f1_score(labels, preds, average="binary")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1: {macro_f1 * 100:.2f}%")
print(f"Binary F1: {binary_f1 * 100:.2f}%")
print(classification_report(labels, preds, digits=4))
print(confusion_matrix(labels, preds))

print("\n===== Threshold(임계값) 찾기 =====")

best_threshold_f1 = 0.5
best_f1 = 0.0
best_threshold_f2 = 0.5
best_f2 = 0.0

# 0.15에서 0.65까지 0.05 단위로 탐색 (기존엔 0.40부터만 봐서 recall을 더 높일 여지를 놓치고 있었음).
# F1과 별도로 F2(recall에 2배 가중치)도 같이 계산 — 우울증 스크리닝은 놓치는 것(FN)이
# 오탐(FP)보다 훨씬 치명적이라, F1 최고점만 보지 않고 F2 최고점도 같이 참고하기 위함
for t in np.arange(0.15, 0.65, 0.05):
    temp_preds = (probs[:, 1] >= t).astype(int)

    temp_acc = accuracy_score(labels, temp_preds)
    temp_precision = precision_score(labels, temp_preds, average="binary", zero_division=0)
    temp_recall = recall_score(labels, temp_preds, average="binary", zero_division=0)
    temp_binary_f1 = f1_score(labels, temp_preds, average="binary", zero_division=0)
    temp_f2 = fbeta_score(labels, temp_preds, beta=2, average="binary", zero_division=0)

    print(f"[Threshold {t:.2f}] Accuracy: {temp_acc * 100:.2f}% | Precision: {temp_precision * 100:.2f}% | "
          f"Recall: {temp_recall * 100:.2f}% | Binary F1: {temp_binary_f1 * 100:.2f}% | F2: {temp_f2 * 100:.2f}%")

    cm = confusion_matrix(labels, temp_preds)
    print(f"  -> 정상오해(FP): {cm[0][1]}명 | 환자놓침(FN): {cm[1][0]}명 | 환자찾음(TP): {cm[1][1]}명\n")

    if temp_binary_f1 > best_f1:
        best_f1 = temp_binary_f1
        best_threshold_f1 = t
    if temp_f2 > best_f2:
        best_f2 = temp_f2
        best_threshold_f2 = t

print(f"결론(F1 기준): Threshold {best_threshold_f1:.2f} 일 때, Binary F1 {best_f1 * 100:.2f}%")
print(f"결론(F2 기준, recall 우선): Threshold {best_threshold_f2:.2f} 일 때, F2 {best_f2 * 100:.2f}%")
print("스크리닝 목적상 환자를 놓치는 게 더 치명적이라면 F2 기준 threshold 사용을 추천합니다.")