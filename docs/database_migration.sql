-- =====================================================
-- 대화 히스토리 관리 시스템 데이터베이스 마이그레이션
-- =====================================================
-- 작성일: 2025-12-04
-- 설명: feedback 테이블에 conversation 관련 컬럼 추가

-- =====================================================
-- Step 1: 기존 feedback 테이블에 컬럼 추가
-- =====================================================

-- conversation_id: 대화 그룹 ID
ALTER TABLE feedback
ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(100);

-- conversation_title: 대화 제목 (첫 메시지에만 저장)
ALTER TABLE feedback
ADD COLUMN IF NOT EXISTS conversation_title TEXT;

-- is_first_message: 대화의 첫 메시지 여부
ALTER TABLE feedback
ADD COLUMN IF NOT EXISTS is_first_message BOOLEAN DEFAULT false;

-- =====================================================
-- Step 2: 인덱스 추가 (성능 최적화)
-- =====================================================

-- conversation_id로 빠른 조회
CREATE INDEX IF NOT EXISTS idx_feedback_conversation
ON feedback(conversation_id);

-- 사용자별 대화 목록 조회 최적화 (생성일 역순)
CREATE INDEX IF NOT EXISTS idx_feedback_user_created
ON feedback(user_id, created_at DESC);

-- 대화별 메시지 조회 최적화 (시간 순)
CREATE INDEX IF NOT EXISTS idx_feedback_conversation_created
ON feedback(conversation_id, created_at ASC);

-- =====================================================
-- Step 3: 기존 데이터 마이그레이션 (선택 사항)
-- =====================================================

-- 기존 데이터에 conversation_id 자동 생성
-- 각 user_id + created_at 기준으로 그룹핑하여 conversation_id 생성

-- 예시: 30분 이내 메시지를 같은 대화로 간주
UPDATE feedback f1
SET conversation_id = CONCAT('conv_', f1.user_id, '_',
    EXTRACT(EPOCH FROM DATE_TRUNC('hour', f1.created_at))::TEXT)
WHERE conversation_id IS NULL;

-- 각 대화의 첫 메시지 표시
WITH first_messages AS (
    SELECT DISTINCT ON (conversation_id)
        uuid,
        conversation_id,
        message
    FROM feedback
    WHERE conversation_id IS NOT NULL
    ORDER BY conversation_id, created_at ASC
)
UPDATE feedback f
SET is_first_message = true,
    conversation_title = CASE
        WHEN LENGTH(fm.message) <= 20 THEN fm.message
        ELSE SUBSTRING(fm.message FROM 1 FOR 17) || '...'
    END
FROM first_messages fm
WHERE f.uuid = fm.uuid;

-- =====================================================
-- Step 4: 제약조건 추가 (선택 사항)
-- =====================================================

-- conversation_id는 NULL 허용 (기존 데이터 호환성)
-- 나중에 NOT NULL로 변경 가능:
-- ALTER TABLE feedback ALTER COLUMN conversation_id SET NOT NULL;

-- =====================================================
-- Step 5: 통계 정보 업데이트
-- =====================================================

ANALYZE feedback;

-- =====================================================
-- 검증 쿼리
-- =====================================================

-- 1. 컬럼 추가 확인
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'feedback'
AND column_name IN ('conversation_id', 'conversation_title', 'is_first_message');

-- 2. 인덱스 생성 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'feedback'
AND indexname LIKE 'idx_feedback_%';

-- 3. 대화 목록 조회 테스트
SELECT
    conversation_id,
    MAX(conversation_title) as title,
    MIN(created_at) as created_at,
    MAX(created_at) as updated_at,
    COUNT(*) as message_count,
    STRING_AGG(
        CASE WHEN is_first_message THEN message ELSE NULL END,
        ''
    ) as first_message
FROM feedback
WHERE user_id = 'test123'  -- 테스트용 user_id
AND conversation_id IS NOT NULL
GROUP BY conversation_id
ORDER BY MAX(created_at) DESC
LIMIT 20;

-- 4. 특정 대화의 메시지 조회 테스트
SELECT
    CASE
        WHEN message IS NOT NULL THEN 'user'
        ELSE 'assistant'
    END as role,
    COALESCE(message, response) as content,
    created_at
FROM feedback
WHERE conversation_id = 'conv_test123_1733285240'  -- 테스트용 conversation_id
ORDER BY created_at ASC;

-- =====================================================
-- 롤백 스크립트 (필요 시 사용)
-- =====================================================

-- 인덱스 삭제
-- DROP INDEX IF EXISTS idx_feedback_conversation;
-- DROP INDEX IF EXISTS idx_feedback_user_created;
-- DROP INDEX IF EXISTS idx_feedback_conversation_created;

-- 컬럼 삭제
-- ALTER TABLE feedback DROP COLUMN IF EXISTS conversation_id;
-- ALTER TABLE feedback DROP COLUMN IF EXISTS conversation_title;
-- ALTER TABLE feedback DROP COLUMN IF EXISTS is_first_message;
