"""
Rate Limiter 구현
- 세션별/IP별 요청 제한
- 슬라이딩 윈도우 방식
- 메모리 기반 (프로덕션에서는 Redis 권장)
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import threading


class RateLimiter:
    """간단한 Rate Limiter (메모리 기반)"""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Args:
            max_requests: 시간 창 내 최대 요청 수
            window_seconds: 시간 창 (초)
        """
        self.max_requests = max_requests
        self.window = timedelta(seconds=window_seconds)
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, identifier: str) -> Tuple[bool, str]:
        """
        요청 허용 여부 확인

        Args:
            identifier: 세션 ID, IP 주소 등

        Returns:
            (허용 여부, 메시지)
        """
        with self.lock:
            now = datetime.now()

            # 오래된 요청 제거 (슬라이딩 윈도우)
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window
            ]

            # 요청 수 확인
            current_count = len(self.requests[identifier])

            if current_count >= self.max_requests:
                remaining_time = int(
                    (self.requests[identifier][0] + self.window - now).total_seconds()
                )
                return False, f"요청 한도 초과. {remaining_time}초 후 다시 시도해주세요."

            # 요청 기록
            self.requests[identifier].append(now)
            return True, "OK"

    def get_stats(self, identifier: str) -> dict:
        """
        특정 식별자의 현재 상태 조회

        Args:
            identifier: 세션 ID, IP 주소 등

        Returns:
            통계 정보 딕셔너리
        """
        with self.lock:
            now = datetime.now()

            # 오래된 요청 제거
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if now - req_time < self.window
            ]

            current_count = len(self.requests[identifier])
            remaining = max(0, self.max_requests - current_count)

            return {
                "current_count": current_count,
                "max_requests": self.max_requests,
                "remaining": remaining,
                "window_seconds": self.window.total_seconds(),
            }

    def cleanup_old_entries(self):
        """메모리 정리 (주기적으로 호출 권장)"""
        with self.lock:
            now = datetime.now()

            # 각 식별자의 오래된 요청 제거
            for identifier in list(self.requests.keys()):
                self.requests[identifier] = [
                    req_time for req_time in self.requests[identifier]
                    if now - req_time < self.window
                ]

                # 비어있는 식별자 제거
                if not self.requests[identifier]:
                    del self.requests[identifier]

    def reset(self, identifier: str = None):
        """
        Rate Limiter 리셋

        Args:
            identifier: 특정 식별자만 리셋 (None이면 전체 리셋)
        """
        with self.lock:
            if identifier:
                if identifier in self.requests:
                    del self.requests[identifier]
            else:
                self.requests.clear()

    def get_all_stats(self) -> dict:
        """모든 식별자의 현재 상태 조회"""
        with self.lock:
            now = datetime.now()

            stats = {}
            for identifier, timestamps in self.requests.items():
                # 오래된 요청 제거
                valid_timestamps = [
                    t for t in timestamps
                    if now - t < self.window
                ]

                if valid_timestamps:
                    stats[identifier] = {
                        "current_count": len(valid_timestamps),
                        "max_requests": self.max_requests,
                        "remaining": max(0, self.max_requests - len(valid_timestamps)),
                    }

            return stats
