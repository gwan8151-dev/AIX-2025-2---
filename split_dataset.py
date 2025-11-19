from pathlib import Path
import random
import shutil

# 0. 설정 부분 -------------------------
VAL_RATIO = 0.2  # 8:2면 0.2

# 라벨링 끝난 이미지/텍스트가 있는 폴더
# 예: my-project/samples/metalcan_nn 이런 식이면 거기 경로 적기
SOURCE_DIR = Path("ai+x_dataset")  # <- 너 폴더명에 맞게 바꿔줘

# 결과를 저장할 루트
OUTPUT_ROOT = Path("Results")
# -------------------------------------


# 1. 이미지+라벨 쌍 모으기
image_paths = [
    p for p in SOURCE_DIR.rglob("*")     # 모든 하위 폴더까지 탐색
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
]


samples = []
for img_path in image_paths:
    lbl_path = img_path.with_suffix(".txt")
    if not lbl_path.exists():
        print(f"[경고] 라벨 없음, 스킵: {img_path.name}")
        continue
    samples.append((img_path, lbl_path))

print(f"총 샘플 개수: {len(samples)}")
print(list(SOURCE_DIR.rglob("*"))[:20])
if len(samples) == 0:
    raise SystemExit("샘플 확인되지 않음")

# 2. 셔플 + 8:2 나누기
random.seed(42)
random.shuffle(samples)

n_val = int(len(samples) * VAL_RATIO)
val_samples = samples[:n_val]
train_samples = samples[n_val:]

print(f"train: {len(train_samples)}, val: {len(val_samples)}")

# 3. 폴더 만들기
for split in ["train", "val"]:
    (OUTPUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

# 4. 복사
def copy_split(split, split_samples):
    for img_path, lbl_path in split_samples:
        dst_img = OUTPUT_ROOT / "images" / split / img_path.name
        dst_lbl = OUTPUT_ROOT / "labels" / split / lbl_path.name

        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)

copy_split("train", train_samples)
copy_split("val", val_samples)
