import os
import shutil

# ============================
# 설정 (여기만 수정)
# ============================
#C:\\Users\\President Eden\\Desktop\\newseed_orig\\seedend\ALL
source_dir = "C:\\Users\\President Eden\\Desktop\\newseed_orig\\seedend\ALL"   # 현재 폴더
target_dir = "C:\\Users\\President Eden\\Desktop\\m_t_d_ver4\\seed_data\\type6"   # 이동할 폴더

# 옮길 번호 리스트
seed_numbers = [88, 6, 54, 124, 71, 68, 145, 72, 137, 23, 62, 58, 63, 48, 81, 112, 103, 86, 138, 67, 4, 51, 38, 95, 83]


# ============================
# 폴더 생성 (없으면)
# ============================
os.makedirs(target_dir, exist_ok=True)


# ============================
# 실행
# ============================
files = os.listdir(source_dir)

moved = 0

for file in files:
    for num in seed_numbers:
        target_name = f"seed ({num})"

        # 파일 이름에 해당 패턴 포함되어 있으면 이동
        if target_name in file:
            src_path = os.path.join(source_dir, file)
            dst_path = os.path.join(target_dir, file)

            shutil.move(src_path, dst_path)
            print(f"[MOVE] {file}")

            moved += 1
            break  # 같은 파일 중복 방지

print(f"\n[OK] 총 이동 파일 수: {moved}")