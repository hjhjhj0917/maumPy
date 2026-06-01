import os
import json
import zipfile
import shutil
from collections import Counter

# =========================================================
# 설정
# =========================================================

DISEASE = "depression"

TRAIN_SOURCE = r"C:\Users\data8316-17\Desktop\융합프로젝트\심리상담 모델\16.심리상담 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터"

TEST_SOURCE = r"C:\Users\data8316-17\Desktop\융합프로젝트\심리상담 모델\16.심리상담 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터"

TRAIN_TARGET = "./data/training"
TEST_TARGET = "./data/test"

# =========================================================
# data 폴더 초기화
# =========================================================

def reset_data_dir():

    if os.path.exists("./data"):
        shutil.rmtree("./data")

    os.makedirs(TRAIN_TARGET, exist_ok=True)
    os.makedirs(TEST_TARGET, exist_ok=True)

    print("\n[INFO] data 폴더 초기화 완료")

# =========================================================
# JSON 검증
# =========================================================

def validate_json(js):

    # disease label 존재 여부
    if DISEASE not in js:
        return False

    label = js.get(DISEASE)

    # label 이상치 제거
    if label not in [0, 1, 2, 3]:
        return False

    # paragraph 존재 여부
    paragraphs = js.get("paragraph")

    if not paragraphs:
        return False

    # paragraph 내용 검사
    valid_sentence_count = 0

    for p in paragraphs:

        text = p.get("paragraph_text", "").strip()

        if len(text) >= 2:
            valid_sentence_count += 1

    # 너무 짧은 상담 제거
    if valid_sentence_count < 3:
        return False

    return True

# =========================================================
# ZIP 처리
# =========================================================

def extract_and_validate_jsons(source_dir, target_dir):

    zip_count = 0
    saved_count = 0
    invalid_count = 0
    duplicate_count = 0

    label_counter = Counter()

    # 중복 체크
    seen_contents = set()

    print(f"\n[{source_dir}] 탐색 시작")

    for root, dirs, files in os.walk(source_dir):

        for file in files:

            if not file.endswith(".zip"):
                continue

            zip_path = os.path.join(root, file)

            zip_count += 1

            try:

                with zipfile.ZipFile(zip_path, "r") as zip_ref:

                    for member in zip_ref.namelist():

                        if (
                            not member.endswith(".json")
                            or member.startswith("__MACOSX")
                        ):
                            continue

                        filename = os.path.basename(member)

                        if not filename:
                            continue

                        try:

                            with zip_ref.open(member) as source:

                                raw_data = source.read()

                                # JSON 파싱
                                js = json.loads(
                                    raw_data.decode("utf-8")
                                )

                                # 데이터 검증
                                if not validate_json(js):
                                    invalid_count += 1
                                    continue

                                # 중복 제거
                                content_signature = json.dumps(
                                    js,
                                    ensure_ascii=False,
                                    sort_keys=True
                                )

                                if content_signature in seen_contents:
                                    duplicate_count += 1
                                    continue

                                seen_contents.add(
                                    content_signature
                                )

                                # 저장 파일명
                                safe_name = (
                                    f"{os.path.splitext(file)[0]}"
                                    f"_{filename}"
                                )

                                target_path = os.path.join(
                                    target_dir,
                                    safe_name
                                )

                                # 저장
                                with open(
                                    target_path,
                                    "w",
                                    encoding="utf-8"
                                ) as out_file:

                                    json.dump(
                                        js,
                                        out_file,
                                        ensure_ascii=False,
                                        indent=2
                                    )

                                saved_count += 1

                                label_counter[
                                    js[DISEASE]
                                ] += 1

                        except Exception:
                            invalid_count += 1

            except zipfile.BadZipFile:

                print(
                    f"[WARNING] 손상된 ZIP 스킵: {zip_path}"
                )

    # 결과 출력
    print("\n==============================")
    print(f"ZIP 파일 수: {zip_count}")
    print(f"저장된 JSON: {saved_count}")
    print(f"중복 제거 수: {duplicate_count}")
    print(f"손상/이상 데이터 제거 수: {invalid_count}")

    print("\n[ 클래스 분포 ]")

    for label in sorted(label_counter.keys()):

        print(
            f"Class {label}: "
            f"{label_counter[label]}"
        )

    print("==============================\n")

# =========================================================
# Train/Test Leakage 검사
# =========================================================

def check_data_leakage():

    train_files = set(os.listdir(TRAIN_TARGET))
    test_files = set(os.listdir(TEST_TARGET))

    overlap = train_files.intersection(test_files)

    print("\n[ Leakage 검사 ]")

    if len(overlap) == 0:
        print("Train/Test 중복 없음")
    else:
        print(f"중복 파일 발견: {len(overlap)}")

# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    print("\n===== 데이터 전처리 시작 =====")

    # 1. data 초기화
    reset_data_dir()

    # 2. Training 처리
    extract_and_validate_jsons(
        TRAIN_SOURCE,
        TRAIN_TARGET
    )

    # 3. Validation 처리
    extract_and_validate_jsons(
        TEST_SOURCE,
        TEST_TARGET
    )

    # 4. Leakage 검사
    check_data_leakage()

    print("\n===== 전처리 완료 =====")
