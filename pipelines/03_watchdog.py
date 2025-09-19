# pipelines/03_watchdog.py
import time
import subprocess
import sys
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import hashlib

DOCS_ROOT = Path("docs")   # 감시할 루트 디렉토리
LOG_FILE = Path("logs/ops.log")
META_PATH = Path("data/artifacts/index_meta.json")

# ✅ fingerprint 계산 함수
def calc_fingerprint(file_path: Path) -> str:
    try:
        m = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                m.update(chunk)
        return m.hexdigest()
    except Exception:
        return None

def load_last_fingerprint():
    if META_PATH.exists():
        try:
            data = json.loads(META_PATH.read_text(encoding='utf-8'))
            # 우선 docs_fingerprint 키를 보고, 구버전 fingerprint 키도 대비
            return data.get('docs_fingerprint') or data.get('fingerprint')
        except Exception:
            return None
    return None

# ✅ 로그 설정 (파일 + 콘솔 동시에 기록)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

class DocsEventHandler(FileSystemEventHandler):
    def __init__(self, debounce_time=5):
        super().__init__()
        self.debounce_time = debounce_time
        self.last_event_time = 0
        self.pending = False

    def on_any_event(self, event):
        if event.is_directory:
            return

        # ✅ 이벤트 타입 필터링 (생성/수정만 허용)
        if event.event_type not in ("modified", "created"):
            return

        # ✅ 파일 확장자 필터링
        if not event.src_path.lower().endswith((".txt", ".pdf")):
            return

        file_path = Path(event.src_path)
        logging.info(f"[WATCHDOG] 변경 감지됨: {file_path} (event={event.event_type})")

        # ✅ fingerprint 비교 (실제 내용이 바뀌었는지 확인)
        new_fp = calc_fingerprint(file_path)
        last_fp = load_last_fingerprint()
        if new_fp and new_fp == last_fp:
            logging.info("[WATCHDOG] fingerprint 동일 → 무시됨 (실제 내용 변경 없음)")
            return

        # 이벤트 발생 시간 갱신
        self.last_event_time = time.time()
        self.pending = True

    def process_pending(self):
        """pending 이벤트가 있으면 debounce 후 파이프라인 실행"""
        if self.pending and (time.time() - self.last_event_time > self.debounce_time):
            logging.info("[WATCHDOG] 여러 파일 이벤트를 모아서 파이프라인 실행 시작")
            try:
                # 1. 인덱스 재생성
                subprocess.run([sys.executable, "-m", "pipelines.01_ingest"], check=True)
                logging.info("[WATCHDOG] 01_ingest 실행 완료 ✅")

                # 2. 평가셋 자동 생성
                subprocess.run([sys.executable, "-m", "pipelines.02_generate_eval"], check=True)
                logging.info("[WATCHDOG] 02_generate_eval 실행 완료 ✅")

                # 3. 평가 실행
                subprocess.run([sys.executable, "-m", "pipelines.02_evaluate"], check=True)
                logging.info("[WATCHDOG] 02_evaluate 실행 완료 ✅")

            except subprocess.CalledProcessError as e:
                logging.error(f"[ERROR] 파이프라인 실행 실패: {e}")
            finally:
                self.pending = False

def main():
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    event_handler = DocsEventHandler(debounce_time=5)
    observer = Observer()
    observer.schedule(event_handler, str(DOCS_ROOT), recursive=True)
    observer.start()

    logging.info("👀 docs/ 디렉토리 감시 시작 (txt, pdf). Ctrl+C로 종료하세요.")
    try:
        while True:
            time.sleep(1)
            event_handler.process_pending()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
