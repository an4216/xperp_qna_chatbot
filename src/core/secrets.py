"""
보안 강화된 비밀 정보 관리 모듈

지원 방식:
1. AWS Secrets Manager (프로덕션 환경 권장)
2. .env 파일 (로컬 개발 환경용)
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# 비밀 정보 사용 모드
SECRETS_MODE = os.getenv("SECRETS_MODE", "env")  # "aws" 또는 "env"
AWS_SECRETS_NAME = os.getenv("AWS_SECRETS_NAME", "inflearn-chatbot/db")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")


class SecretsManager:
    """비밀 정보 관리 클래스"""

    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        self.mode = SECRETS_MODE

    def get_db_config(self) -> Dict[str, Any]:
        """
        데이터베이스 연결 정보 가져오기

        Returns:
            DB 설정 딕셔너리 {host, port, database, user, password}
        """
        if self._cache:
            return self._cache

        if self.mode == "aws":
            config = self._get_from_aws_secrets_manager()
        else:
            config = self._get_from_env()

        self._cache = config
        return config

    def _get_from_aws_secrets_manager(self) -> Dict[str, Any]:
        """
        AWS Secrets Manager에서 비밀 정보 가져오기

        Returns:
            DB 설정 딕셔너리
        """
        try:
            import boto3
            from botocore.exceptions import ClientError

            # Secrets Manager 클라이언트 생성
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=AWS_REGION
            )

            # 비밀 정보 가져오기
            response = client.get_secret_value(SecretId=AWS_SECRETS_NAME)

            # JSON 파싱
            if 'SecretString' in response:
                secret = json.loads(response['SecretString'])
            else:
                # 바이너리 비밀 정보 (사용 안 함)
                raise ValueError("Binary secrets not supported")

            print(f"[INFO] AWS Secrets Manager에서 DB 정보 로드 완료 ({AWS_SECRETS_NAME})")

            # 필수 필드 검증
            required_fields = ['host', 'port', 'database', 'username', 'password']
            for field in required_fields:
                if field not in secret:
                    raise ValueError(f"Missing required field: {field}")

            return {
                'host': secret['host'],
                'port': int(secret['port']),
                'database': secret['database'],
                'user': secret['username'],  # AWS는 username 사용
                'password': secret['password']
            }

        except ImportError:
            print("[ERROR] boto3가 설치되지 않았습니다. pip install boto3 실행")
            print("[INFO] .env 파일로 폴백합니다.")
            return self._get_from_env()

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                print(f"[ERROR] AWS Secrets Manager에서 비밀 정보를 찾을 수 없습니다: {AWS_SECRETS_NAME}")
            elif error_code == 'AccessDeniedException':
                print(f"[ERROR] AWS Secrets Manager 접근 권한이 없습니다")
            else:
                print(f"[ERROR] AWS Secrets Manager 오류: {e}")

            print("[INFO] .env 파일로 폴백합니다.")
            return self._get_from_env()

        except Exception as e:
            print(f"[ERROR] AWS Secrets Manager에서 비밀 정보 로드 실패: {e}")
            print("[INFO] .env 파일로 폴백합니다.")
            return self._get_from_env()

    def _get_from_env(self) -> Dict[str, Any]:
        """
        .env 파일에서 비밀 정보 가져오기

        Returns:
            DB 설정 딕셔너리
        """
        config = {
            'host': os.getenv("DB_HOST", "localhost"),
            'port': int(os.getenv("DB_PORT", "7000")),
            'database': os.getenv("DB_NAME", "postgres"),
            'user': os.getenv("DB_USER", "postgres"),
            'password': os.getenv("DB_PASSWORD", "")
        }

        if not config['password']:
            print("[WARN] DB_PASSWORD가 설정되지 않았습니다!")

        return config

    def clear_cache(self):
        """캐시 초기화 (비밀 정보 갱신 시 사용)"""
        self._cache = None


# 싱글톤 인스턴스
_secrets_manager = SecretsManager()


def get_db_config() -> Dict[str, Any]:
    """
    데이터베이스 연결 정보 가져오기 (전역 함수)

    Returns:
        DB 설정 딕셔너리

    Example:
        >>> config = get_db_config()
        >>> print(config['host'], config['port'])
    """
    return _secrets_manager.get_db_config()


def clear_secrets_cache():
    """비밀 정보 캐시 초기화 (전역 함수)"""
    _secrets_manager.clear_cache()
