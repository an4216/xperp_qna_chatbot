"""
PostgreSQL 데이터베이스 연결 모듈

이 모듈은 PostgreSQL 데이터베이스 연결 및 기본 쿼리 작업을 담당합니다.
- 연결 풀 관리
- 연결 상태 확인
- 기본 쿼리 실행 (SELECT, INSERT, UPDATE, DELETE)
"""

import os
import psycopg2
from psycopg2 import pool, sql, Error
from psycopg2.extensions import connection as Connection
from dotenv import load_dotenv
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from src.core.secrets import get_db_config

# 환경변수 로드
load_dotenv()

# ✅ 보안 강화: Secrets Manager에서 DB 정보 가져오기
_db_config = get_db_config()
DB_HOST = _db_config['host']
DB_PORT = _db_config['port']
DB_NAME = _db_config['database']
DB_USER = _db_config['user']
DB_PASSWORD = _db_config['password']

# 연결 풀 (전역 변수)
_connection_pool: Optional[pool.SimpleConnectionPool] = None


# =========================================
# 1. 연결 풀 관리
# =========================================
def get_connection_pool() -> Optional[pool.SimpleConnectionPool]:
    """
    PostgreSQL 연결 풀 반환 (싱글톤 패턴)

    Returns:
        psycopg2 연결 풀 객체 (연결 실패 시 None)
    """
    global _connection_pool

    if _connection_pool is None:
        try:
            _connection_pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=DB_HOST,
                port=DB_PORT,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            print(f"[INFO] PostgreSQL 연결 풀 생성 완료 ({DB_HOST}:{DB_PORT}/{DB_NAME})")
        except Error as e:
            print(f"[WARN] PostgreSQL 연결 실패 (DB 기능 비활성화됨): {e}")
            print(f"[INFO] 챗봇은 정상 작동하며, 피드백 저장만 비활성화됩니다.")
            return None

    return _connection_pool


@contextmanager
def get_db_connection():
    """
    데이터베이스 연결을 컨텍스트 매니저로 제공

    사용 예시:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users")
            results = cursor.fetchall()

    Yields:
        psycopg2 Connection 객체 (연결 실패 시 None)
    """
    connection_pool = get_connection_pool()

    # 연결 풀이 없으면 None 반환 (DB 비활성화 상태)
    if connection_pool is None:
        yield None
        return

    conn = None

    try:
        conn = connection_pool.getconn()
        yield conn
    except Error as e:
        if conn:
            conn.rollback()
        print(f"[ERROR] 데이터베이스 작업 중 오류 발생: {e}")
        raise
    finally:
        if conn:
            connection_pool.putconn(conn)


def close_connection_pool():
    """
    연결 풀 종료 (애플리케이션 종료 시 호출)
    """
    global _connection_pool

    if _connection_pool:
        _connection_pool.closeall()
        _connection_pool = None
        print("[INFO] PostgreSQL 연결 풀 종료 완료")


# =========================================
# 2. 연결 상태 확인
# =========================================
def test_connection() -> Tuple[bool, str]:
    """
    데이터베이스 연결 테스트

    Returns:
        (성공 여부, 메시지)
    """
    try:
        with get_db_connection() as conn:
            # DB가 비활성화된 경우
            if conn is None:
                error_message = (
                    f"[DISABLED] PostgreSQL 연결 비활성화됨\n"
                    f"   - 호스트: {DB_HOST}:{DB_PORT}\n"
                    f"   - 데이터베이스: {DB_NAME}\n"
                    f"   - 상태: DB 없이 서비스 가능"
                )
                return False, error_message

            cursor = conn.cursor()

            # PostgreSQL 버전 확인
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]

            # 현재 데이터베이스 확인
            cursor.execute("SELECT current_database();")
            current_db = cursor.fetchone()[0]

            # 연결 정보
            cursor.execute("SELECT current_user;")
            current_user = cursor.fetchone()[0]

            cursor.close()

            message = (
                f"[SUCCESS] PostgreSQL 연결 성공!\n"
                f"   - 호스트: {DB_HOST}:{DB_PORT}\n"
                f"   - 데이터베이스: {current_db}\n"
                f"   - 사용자: {current_user}\n"
                f"   - 버전: {version.split(',')[0]}"
            )

            return True, message

    except Error as e:
        error_message = (
            f"[FAILED] PostgreSQL 연결 실패!\n"
            f"   - 호스트: {DB_HOST}:{DB_PORT}\n"
            f"   - 데이터베이스: {DB_NAME}\n"
            f"   - 오류: {str(e)}"
        )
        return False, error_message


# =========================================
# 3. 기본 쿼리 실행
# =========================================
def execute_query(query: str, params: Optional[Tuple] = None) -> bool:
    """
    INSERT, UPDATE, DELETE 쿼리 실행

    Args:
        query: SQL 쿼리
        params: 쿼리 파라미터 (옵션)

    Returns:
        성공 여부
    """
    try:
        with get_db_connection() as conn:
            # DB가 비활성화된 경우
            if conn is None:
                print(f"[WARN] DB 비활성화 상태 - 쿼리 실행 생략")
                return False

            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            cursor.close()
            return True
    except Error as e:
        print(f"[ERROR] 쿼리 실행 실패: {e}")
        return False


def fetch_one(query: str, params: Optional[Tuple] = None) -> Optional[Tuple]:
    """
    SELECT 쿼리 실행 (단일 결과 반환)
    INSERT/UPDATE/DELETE with RETURNING 절도 지원

    Args:
        query: SQL 쿼리
        params: 쿼리 파라미터 (옵션)

    Returns:
        쿼리 결과 (단일 행)
    """
    try:
        with get_db_connection() as conn:
            # DB가 비활성화된 경우
            if conn is None:
                print(f"[WARN] DB 비활성화 상태 - 쿼리 실행 생략")
                return None

            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()

            # INSERT/UPDATE/DELETE (RETURNING 포함) 시 커밋 필요
            query_upper = query.strip().upper()
            if query_upper.startswith(('INSERT', 'UPDATE', 'DELETE')):
                conn.commit()

            cursor.close()
            return result
    except Error as e:
        print(f"[ERROR] 쿼리 실행 실패: {e}")
        return None


def fetch_all(query: str, params: Optional[Tuple] = None) -> List[Tuple]:
    """
    SELECT 쿼리 실행 (전체 결과 반환)

    Args:
        query: SQL 쿼리
        params: 쿼리 파라미터 (옵션)

    Returns:
        쿼리 결과 (전체 행)
    """
    try:
        with get_db_connection() as conn:
            # DB가 비활성화된 경우
            if conn is None:
                print(f"[WARN] DB 비활성화 상태 - 쿼리 실행 생략")
                return []

            cursor = conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
    except Error as e:
        print(f"[ERROR] 쿼리 실행 실패: {e}")
        return []


def fetch_all_dict(query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """
    SELECT 쿼리 실행 (딕셔너리 형태로 반환)

    Args:
        query: SQL 쿼리
        params: 쿼리 파라미터 (옵션)

    Returns:
        쿼리 결과 (딕셔너리 리스트)
    """
    try:
        with get_db_connection() as conn:
            # DB가 비활성화된 경우
            if conn is None:
                print(f"[WARN] DB 비활성화 상태 - 쿼리 실행 생략")
                return []

            cursor = conn.cursor()
            cursor.execute(query, params)

            # 컬럼명 추출
            columns = [desc[0] for desc in cursor.description]

            # 결과를 딕셔너리로 변환
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))

            cursor.close()
            return results
    except Error as e:
        print(f"[ERROR] 쿼리 실행 실패: {e}")
        return []


# =========================================
# 4. 테이블 존재 여부 확인
# =========================================
def table_exists(table_name: str) -> bool:
    """
    테이블 존재 여부 확인

    Args:
        table_name: 테이블명

    Returns:
        존재 여부
    """
    # DB가 비활성화된 경우
    connection_pool = get_connection_pool()
    if connection_pool is None:
        return False

    query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = %s
        );
    """
    result = fetch_one(query, (table_name,))
    return result[0] if result else False


# =========================================
# 5. 데이터베이스 정보 조회
# =========================================
def get_database_info() -> Dict[str, Any]:
    """
    데이터베이스 기본 정보 조회

    Returns:
        데이터베이스 정보 딕셔너리
    """
    info = {
        "host": DB_HOST,
        "port": DB_PORT,
        "database": DB_NAME,
        "user": DB_USER,
        "tables": [],
        "table_count": 0,
        "connected": False
    }

    try:
        with get_db_connection() as conn:
            # DB가 비활성화된 경우
            if conn is None:
                info["connected"] = False
                return info

            info["connected"] = True
            cursor = conn.cursor()

            # 테이블 목록 조회
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)

            tables = [row[0] for row in cursor.fetchall()]
            info["tables"] = tables
            info["table_count"] = len(tables)

            cursor.close()
    except Error as e:
        print(f"[ERROR] 데이터베이스 정보 조회 실패: {e}")

    return info
