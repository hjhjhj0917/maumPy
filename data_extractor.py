import os
import zipfile
import shutil


def extract_and_move_jsons(source_dir, target_dir):
    """
    여러 ZIP 파일을 탐색하여 JSON 파일만 추출한 뒤 target_dir로 이동하는 함수
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    zip_count = 0
    json_count = 0

    print(f"🔍 [{source_dir}] 탐색 시작...")

    # source_dir 하위의 모든 파일 탐색
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                zip_count += 1

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            # 맥 OS 압축 잔여물이나 폴더 제외하고 순수 .json 파일만 타겟
                            if member.endswith('.json') and not member.startswith('__MACOSX'):
                                # 파일 이름만 추출
                                filename = os.path.basename(member)
                                if not filename: continue

                                # 중복 방지를 위해 [압축파일명_원래파일.json] 형태로 이름 변경
                                safe_name = f"{os.path.splitext(file)[0]}_{filename}"
                                target_path = os.path.join(target_dir, safe_name)

                                # 압축 해제 및 이동
                                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                    shutil.copyfileobj(source, target)
                                json_count += 1

                except zipfile.BadZipFile:
                    print(f"⚠️ 손상된 ZIP 파일 스킵: {zip_path}")

    print(f"✅ 완료! {zip_count}개의 ZIP 파일에서 {json_count}개의 JSON 파일을 {target_dir}로 복사했습니다.\n")


if __name__ == "__main__":
    # 아까 준모님이 알려주신 원본 데이터 경로를 넣어주세요. (경로 앞에 r을 붙이면 역슬래시 오류가 안 납니다)

    # 1. Training (학습 데이터) 추출 -> data/training 폴더로
    train_source = r"C:\Users\data8316-17\Desktop\융합프로젝트\심리상담 모델\16.심리상담 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터"
    train_target = "./data/training"
    extract_and_move_jsons(train_source, train_target)

    # 2. Validation (테스트 데이터) 추출 -> data/test 폴더로
    val_source = r"C:\Users\data8316-17\Desktop\융합프로젝트\심리상담 모델\16.심리상담 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터"
    val_target = "./data/test"
    extract_and_move_jsons(val_source, val_target)