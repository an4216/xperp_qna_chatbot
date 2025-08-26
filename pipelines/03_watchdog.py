# pipelines/03_watchdog.py
import time
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DOCS_ROOT = Path("docs")  # 감시할 루트 디렉토리

class DocsEventHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if event.is_directory:
            return
        # txt/pdf 파일만 감지
        if not event.src_path.lower().endswith((".txt", ".pdf")):
            return

        print(f"\n[WATCHDOG] 변경 감지됨: {event.src_path}")
        try:
            # 1. 인덱스 재생성
            subprocess.run([sys.executable, "-m", "pipelines.01_ingest"], check=True)
            # 2. 평가 실행
            subprocess.run([sys.executable, "-m", "pipelines.02_evaluate"], check=True)
            print("[WATCHDOG] 인덱싱 및 평가 완료 ✅")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] 파이프라인 실행 실패: {e}")

def main():
    DOCS_ROOT.mkdir(parents=True, exist_ok=True)
    event_handler = DocsEventHandler()
    observer = Observer()
    observer.schedule(event_handler, str(DOCS_ROOT), recursive=True)
    observer.start()

    print("👀 docs/ 디렉토리 감시 시작 (txt, pdf). Ctrl+C로 종료하세요.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
