# 데이터베이스 재생성 가이드

현재 DB를 삭제하고 새로운 스키마로 재생성하는 방법

---

## ⚠️ 주의사항

**이 작업은 기존 데이터를 모두 삭제합니다!**
- 프로덕션 환경에서는 반드시 백업 후 실행하세요
- 테스트 환경에서 먼저 검증하세요

---

## 📋 사전 준비

### 1. PostgreSQL 접속 정보 확인

`.env` 파일에서 DB 접속 정보 확인:
```bash
DB_HOST=localhost
DB_PORT=7000
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=AegisAI7240!!
```

### 2. PostgreSQL 클라이언트 설치 확인

```bash
# PostgreSQL 버전 확인
psql --version

# 출력 예시: psql (PostgreSQL) 14.5
```

---

## 🚀 실행 방법

### Option 1: psql 명령어로 실행 (권장)

```bash
# 1. PostgreSQL 접속
psql -h localhost -p 7000 -U postgres -d postgres

# 비밀번호 입력: AegisAI7240!!

# 2. 스크립트 실행
\i docs/database_recreate.sql

# 3. 종료
\q
```

### Option 2: 한 줄 명령어로 실행

**Windows (PowerShell)**:
```powershell
$env:PGPASSWORD="AegisAI7240!!"
psql -h localhost -p 7000 -U postgres -d postgres -f docs/database_recreate.sql
```

**Windows (CMD)**:
```cmd
set PGPASSWORD=AegisAI7240!!
psql -h localhost -p 7000 -U postgres -d postgres -f docs/database_recreate.sql
```

**Linux/Mac**:
```bash
PGPASSWORD='AegisAI7240!!' psql -h localhost -p 7000 -U postgres -d postgres -f docs/database_recreate.sql
```

### Option 3: Docker로 실행 (Docker 사용 시)

```bash
docker exec -i postgres_container psql -U postgres -d postgres < docs/database_recreate.sql
```

---

## 🔍 실행 결과 확인

### 1. 테이블 생성 확인

```sql
-- PostgreSQL 접속 후
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name = 'feedback';

-- 결과: feedback
```

### 2. 컬럼 확인

```sql
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'feedback'
ORDER BY ordinal_position;
```

**예상 결과**:
```
column_name          | data_type
---------------------+------------------------
uuid                 | uuid
user_id              | character varying
conversation_id      | character varying
conversation_title   | text
is_first_message     | boolean
message              | text
response             | text
created_at           | timestamp without time zone
feedback             | character varying
reason               | text
comment              | text
name                 | character varying
timestamp            | timestamp without time zone
feedback_updated_at  | timestamp without time zone
```

### 3. 인덱스 확인

```sql
SELECT indexname
FROM pg_indexes
WHERE tablename = 'feedback';
```

**예상 결과**:
```
indexname
---------------------------------------
feedback_pkey
idx_feedback_user_created
idx_feedback_conversation
idx_feedback_conversation_created
idx_feedback_first_message
```

### 4. 테스트 데이터 확인

```sql
SELECT
    COUNT(*) as total_messages,
    COUNT(DISTINCT user_id) as total_users,
    COUNT(DISTINCT conversation_id) as total_conversations
FROM feedback;
```

**예상 결과**:
```
total_messages | total_users | total_conversations
---------------+-------------+--------------------
            6  |           2 |                  3
```

### 5. 대화 목록 조회 테스트

```sql
SELECT
    conversation_id,
    MAX(conversation_title) as title,
    COUNT(*) as message_count,
    MIN(created_at) as created_at,
    MAX(created_at) as updated_at
FROM feedback
WHERE user_id = 'test123'
GROUP BY conversation_id
ORDER BY MAX(created_at) DESC;
```

**예상 결과**:
```
conversation_id              | title                  | message_count | created_at          | updated_at
-----------------------------+------------------------+---------------+---------------------+---------------------
conv_test123_1733285000000   | 부양가족 등록 방법      |             2 | 2025-12-03 14:00:00 | 2025-12-03 14:01:00
conv_test123_1733280000000   | 연말정산 세액공제 문의  |             3 | 2025-12-03 10:00:00 | 2025-12-03 10:02:00
```

---

## 🛠️ 문제 해결

### 1. "psql: command not found"

**원인**: PostgreSQL 클라이언트가 설치되지 않음

**해결**:
```bash
# Windows (Chocolatey)
choco install postgresql

# Mac (Homebrew)
brew install postgresql

# Ubuntu/Debian
sudo apt-get install postgresql-client
```

### 2. "connection refused"

**원인**: PostgreSQL 서버가 실행 중이지 않음

**해결**:
```bash
# PostgreSQL 서버 상태 확인
# Windows (서비스 관리자)
services.msc

# Linux/Mac
sudo systemctl status postgresql
```

### 3. "password authentication failed"

**원인**: 비밀번호가 틀림

**해결**:
- `.env` 파일에서 `DB_PASSWORD` 확인
- 환경변수가 제대로 설정되었는지 확인

### 4. "database does not exist"

**원인**: 지정한 데이터베이스가 없음

**해결**:
```bash
# PostgreSQL 접속 (기본 postgres DB)
psql -h localhost -p 7000 -U postgres

# 데이터베이스 생성
CREATE DATABASE postgres;

# 종료 후 다시 실행
\q
```

### 5. "permission denied"

**원인**: 사용자 권한 부족

**해결**:
```bash
# 슈퍼유저로 접속
psql -h localhost -p 7000 -U postgres -d postgres

# 사용자 권한 확인
\du

# 권한 부여 (필요 시)
ALTER USER postgres CREATEDB;
```

---

## 📊 스키마 설명

### feedback 테이블 구조

```sql
CREATE TABLE feedback (
    -- 기본 필드
    uuid UUID PRIMARY KEY,              -- 메시지 고유 ID
    user_id VARCHAR(50),                -- 사용자 ID

    -- 대화 관리 (NEW)
    conversation_id VARCHAR(100),       -- 대화 그룹 ID
    conversation_title TEXT,            -- 대화 제목 (첫 메시지에만)
    is_first_message BOOLEAN,           -- 첫 메시지 플래그

    -- 메시지
    message TEXT,                       -- 사용자 질문
    response TEXT,                      -- AI 응답

    -- 시간
    created_at TIMESTAMP,               -- 생성 시간

    -- 피드백
    feedback VARCHAR(10),               -- 'up' or 'down'
    reason TEXT,                        -- 피드백 이유
    comment TEXT,                       -- 피드백 코멘트
    name VARCHAR(100),                  -- 피드백 제공자 이름
    timestamp TIMESTAMP,                -- 피드백 시간
    feedback_updated_at TIMESTAMP       -- 피드백 수정 시간
);
```

### 인덱스 설명

```sql
-- 1. 사용자별 대화 목록 조회 (가장 많이 사용)
idx_feedback_user_created: (user_id, created_at DESC)

-- 2. conversation_id로 조회
idx_feedback_conversation: (conversation_id)

-- 3. 대화별 메시지 조회 (시간 순)
idx_feedback_conversation_created: (conversation_id, created_at ASC)

-- 4. 첫 메시지만 조회 (대화 제목)
idx_feedback_first_message: (conversation_id, is_first_message)
WHERE is_first_message = true
```

---

## 🔄 백업 및 복구

### 백업 방법

**1. 전체 백업 (SQL)**:
```bash
pg_dump -h localhost -p 7000 -U postgres -d postgres -t feedback > feedback_backup.sql
```

**2. CSV 백업**:
```sql
-- PostgreSQL 접속 후
COPY feedback TO '/tmp/feedback_backup.csv' WITH CSV HEADER;
```

**3. 백업 테이블 생성**:
```sql
CREATE TABLE feedback_backup AS SELECT * FROM feedback;
```

### 복구 방법

**1. SQL 복구**:
```bash
psql -h localhost -p 7000 -U postgres -d postgres < feedback_backup.sql
```

**2. CSV 복구**:
```sql
COPY feedback FROM '/tmp/feedback_backup.csv' WITH CSV HEADER;
```

**3. 백업 테이블에서 복구**:
```sql
INSERT INTO feedback SELECT * FROM feedback_backup;
```

---

## ✅ 다음 단계

데이터베이스 재생성 완료 후:

1. **서버 시작 테스트**
   ```bash
   uvicorn main:app --reload
   ```

2. **헬스 체크**
   ```bash
   curl http://localhost:8000/health
   ```

3. **챗봇 접속**
   ```
   http://localhost:8000/?userId=test123
   ```

4. **테스트 메시지 전송**
   - "연말정산이 뭐야?" 입력
   - DB에 저장되는지 확인

5. **DB 확인**
   ```sql
   SELECT * FROM feedback ORDER BY created_at DESC LIMIT 5;
   ```

---

## 📞 문제 발생 시

1. 로그 확인
   ```bash
   # 서버 로그 확인
   tail -f server.log
   ```

2. DB 연결 테스트
   ```python
   # test_db_connection.py
   from src.core.database import test_connection

   success, message = test_connection()
   print(message)
   ```

3. 이슈 리포트
   - 에러 메시지 전체 복사
   - `.env` 설정 확인 (비밀번호 제외)
   - PostgreSQL 버전 확인

---

## 📚 참고 자료

- PostgreSQL 공식 문서: https://www.postgresql.org/docs/
- psql 명령어: https://www.postgresql.org/docs/current/app-psql.html
- pg_dump 백업: https://www.postgresql.org/docs/current/app-pgdump.html
