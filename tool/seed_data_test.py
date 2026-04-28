import re
from pathlib import Path

# =========================================
# 설정
# =========================================
TARGET_DIR = Path(".")         # 현재 폴더
RECURSIVE = False              # 하위 폴더까지 검사하려면 True
MAX_SHOW_PER_FILE = 20         # 파일별 표시할 오류 개수 제한


# =========================================
# 숫자 1개만 있는 줄인지 판정
# 허용 예:
#   1
#   -2
#   3.14
#   -3.2e-04
#   +5.0E+03
# =========================================
NUM_PATTERN = re.compile(
    r'^[\+\-]?('
    r'(\d+(\.\d*)?)'           # 1, 1., 1.23
    r'|'
    r'(\.\d+)'                 # .25
    r')([eE][\+\-]?\d+)?$'     # optional exponent
)


def is_single_numeric_token(line: str) -> tuple[bool, str]:
    s = line.strip()

    # 빈 줄
    if s == "":
        return False, "빈 줄"

    # 공백 기준 분리
    parts = s.split()

    # 2열 이상
    if len(parts) != 1:
        return False, f"{len(parts)}열 데이터"

    token = parts[0]

    # 숫자 형식 검사
    if not NUM_PATTERN.match(token):
        return False, f"숫자 아님: {token}"

    return True, ""


def inspect_txt_file(path: Path) -> list[tuple[int, str, str]]:
    errors = []

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                ok, reason = is_single_numeric_token(line)
                if not ok:
                    errors.append((lineno, reason, line.rstrip("\n")))
    except Exception as e:
        errors.append((0, f"파일 읽기 실패: {e}", ""))

    return errors


def main():
    txt_files = sorted(TARGET_DIR.rglob("*.txt") if RECURSIVE else TARGET_DIR.glob("*.txt"))

    if not txt_files:
        print(f"[INFO] .txt 파일이 없습니다: {TARGET_DIR.resolve()}")
        return

    total_files = len(txt_files)
    ok_files = 0
    bad_files = 0

    print(f"[검사 시작] 폴더: {TARGET_DIR.resolve()}")
    print(f"[대상 파일 수] {total_files}")
    print("-" * 80)

    for path in txt_files:
        errors = inspect_txt_file(path)

        if not errors:
            ok_files += 1
            print(f"[OK] {path.name}")
        else:
            bad_files += 1
            print(f"[NG] {path.name}  -> 문제 {len(errors)}개")

            for i, (lineno, reason, raw) in enumerate(errors[:MAX_SHOW_PER_FILE], start=1):
                if lineno == 0:
                    print(f"   - {reason}")
                else:
                    print(f"   - line {lineno}: {reason}")
                    print(f"     content: {raw}")

            if len(errors) > MAX_SHOW_PER_FILE:
                print(f"   ... 추가 오류 {len(errors) - MAX_SHOW_PER_FILE}개 생략")

        print("-" * 80)

    print("[검사 완료]")
    print(f"정상 파일: {ok_files}")
    print(f"문제 파일: {bad_files}")
    print(f"전체 파일: {total_files}")


if __name__ == "__main__":
    main()