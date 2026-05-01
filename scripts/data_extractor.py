import os
import zipfile
import shutil


def extract_and_move_jsons(source_dir, target_dir):

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    zip_count = 0
    json_count = 0

    print(f"[{source_dir}] 탐색 시작...")

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                zip_count += 1

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            if member.endswith('.json') and not member.startswith('__MACOSX'):
                                filename = os.path.basename(member)
                                if not filename: continue

                                safe_name = f"{os.path.splitext(file)[0]}_{filename}"
                                target_path = os.path.join(target_dir, safe_name)

                                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                                    shutil.copyfileobj(source, target)
                                json_count += 1

                except zipfile.BadZipFile:
                    print(f"손상된 ZIP 파일 스킵: {zip_path}")

    print(f"완료! {zip_count}개의 ZIP 파일에서 {json_count}개의 JSON 파일을 {target_dir}로 복사했습니다.\n")


if __name__ == "__main__":

    train_source = r"C:\Users\data8316-17\Desktop\융합프로젝트\심리상담 모델\16.심리상담 데이터\3.개방데이터\1.데이터\Training\02.라벨링데이터"
    train_target = "./data/training"
    extract_and_move_jsons(train_source, train_target)

    val_source = r"C:\Users\data8316-17\Desktop\융합프로젝트\심리상담 모델\16.심리상담 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터"
    val_target = "./data/test"
    extract_and_move_jsons(val_source, val_target)