"""
AWS Secrets Manager에 DB 연결 정보 저장 스크립트

실행 전 필요사항:
1. AWS CLI 설치 및 구성: aws configure
2. boto3 설치: pip install boto3
3. IAM 권한: secretsmanager:CreateSecret, secretsmanager:PutSecretValue
"""

import json
import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

# AWS 설정
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
SECRET_NAME = os.getenv("AWS_SECRETS_NAME", "inflearn-chatbot/db")

# DB 정보 (현재 .env에서 읽기)
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "7000")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

print("=" * 80)
print("AWS Secrets Manager에 DB 연결 정보 저장")
print("=" * 80)
print()

if not DB_PASSWORD:
    print("[ERROR] DB_PASSWORD가 설정되지 않았습니다!")
    print("        .env 파일에 DB_PASSWORD를 설정해주세요.")
    exit(1)

# 비밀 정보 JSON
secret_data = {
    "host": DB_HOST,
    "port": DB_PORT,
    "database": DB_NAME,
    "username": DB_USER,
    "password": DB_PASSWORD,
    "engine": "postgres"
}

print(f"저장할 비밀 정보:")
print(f"  - Secret Name: {SECRET_NAME}")
print(f"  - Region: {AWS_REGION}")
print(f"  - Host: {DB_HOST}")
print(f"  - Port: {DB_PORT}")
print(f"  - Database: {DB_NAME}")
print(f"  - Username: {DB_USER}")
print(f"  - Password: {'*' * len(DB_PASSWORD)}")
print()

# 확인
response = input("AWS Secrets Manager에 저장하시겠습니까? (yes/no): ")
if response.lower() != 'yes':
    print("취소되었습니다.")
    exit(0)

try:
    # Secrets Manager 클라이언트 생성
    client = boto3.client(
        service_name='secretsmanager',
        region_name=AWS_REGION
    )

    # 비밀 정보 생성
    try:
        response = client.create_secret(
            Name=SECRET_NAME,
            Description='Inflearn Chatbot Database Credentials',
            SecretString=json.dumps(secret_data),
            Tags=[
                {'Key': 'Project', 'Value': 'Inflearn-Chatbot'},
                {'Key': 'Environment', 'Value': 'Production'},
                {'Key': 'ManagedBy', 'Value': 'Python-Script'}
            ]
        )
        print()
        print(f"[SUCCESS] 비밀 정보가 생성되었습니다!")
        print(f"  ARN: {response['ARN']}")
        print(f"  Name: {response['Name']}")

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceExistsException':
            # 이미 존재하면 업데이트
            print()
            print(f"[INFO] 비밀 정보가 이미 존재합니다. 업데이트합니다...")
            response = client.put_secret_value(
                SecretId=SECRET_NAME,
                SecretString=json.dumps(secret_data)
            )
            print(f"[SUCCESS] 비밀 정보가 업데이트되었습니다!")
            print(f"  ARN: {response['ARN']}")
            print(f"  Name: {response['Name']}")
            print(f"  Version: {response['VersionId']}")
        else:
            raise

    print()
    print("=" * 80)
    print("다음 단계:")
    print("=" * 80)
    print()
    print("1. .env 파일에 다음 설정 추가:")
    print(f"   SECRETS_MODE=aws")
    print(f"   AWS_SECRETS_NAME={SECRET_NAME}")
    print(f"   AWS_REGION={AWS_REGION}")
    print()
    print("2. .env 파일에서 DB 비밀번호 제거 (선택사항):")
    print("   # DB_PASSWORD=...  <- 주석 처리 또는 삭제")
    print()
    print("3. IAM 역할/사용자에 권한 추가:")
    print("   - secretsmanager:GetSecretValue")
    print(f"   - Resource: arn:aws:secretsmanager:{AWS_REGION}:*:secret:{SECRET_NAME}*")
    print()
    print("4. 애플리케이션 재시작")
    print()

except ClientError as e:
    error_code = e.response['Error']['Code']
    print()
    print(f"[ERROR] AWS Secrets Manager 오류: {error_code}")
    print(f"  메시지: {e.response['Error']['Message']}")
    print()

    if error_code == 'AccessDeniedException':
        print("해결 방법:")
        print("  1. AWS CLI 설정 확인: aws configure")
        print("  2. IAM 권한 확인:")
        print("     - secretsmanager:CreateSecret")
        print("     - secretsmanager:PutSecretValue")
    elif error_code == 'InvalidRequestException':
        print("해결 방법:")
        print("  1. Secret Name 형식 확인")
        print("  2. Region 설정 확인")

except Exception as e:
    print()
    print(f"[ERROR] 예기치 못한 오류: {e}")
    import traceback
    traceback.print_exc()
