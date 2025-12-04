-- =====================================================
-- 데이터베이스 완전 재생성 스크립트
-- =====================================================
-- 작성일: 2025-12-04
-- 경고: 이 스크립트는 기존 데이터를 모두 삭제합니다!
-- 프로덕션 환경에서는 반드시 백업 후 실행하세요!

-- =====================================================
-- Step 0: 기존 데이터 백업 (선택 사항)
-- =====================================================

-- 백업 테이블 생성 (기존 데이터 보존)
-- DROP TABLE IF EXISTS feedback_backup;
-- CREATE TABLE feedback_backup AS SELECT * FROM feedback;

-- CSV로 백업
-- COPY feedback TO '/tmp/feedback_backup.csv' WITH CSV HEADER;

-- =====================================================
-- Step 1: 기존 테이블 삭제
-- =====================================================

-- 외래키 제약조건이 있다면 먼저 삭제
-- (현재는 없지만, 나중을 위해 포함)
DROP TABLE IF EXISTS feedback CASCADE;

-- =====================================================
-- Step 2: 새 테이블 생성 (conversation 기능 포함)
-- =====================================================

CREATE TABLE feedback (
    -- 기본 필드
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(50) NOT NULL,

    -- 대화 관리 필드 (NEW)
    conversation_id VARCHAR(100),
    conversation_title TEXT,
    is_first_message BOOLEAN DEFAULT false,

    -- 메시지 내용
    message TEXT,
    response TEXT,

    -- 시간 정보
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- 피드백 정보
    feedback VARCHAR(10),  -- 'up' 또는 'down'
    reason TEXT,
    comment TEXT,
    name VARCHAR(100),

    -- 피드백 시간
    timestamp TIMESTAMP,
    feedback_updated_at TIMESTAMP
);

-- =====================================================
-- Step 3: 인덱스 생성 (성능 최적화)
-- =====================================================

-- 1. 사용자별 대화 목록 조회 최적화 (가장 중요)
CREATE INDEX idx_feedback_user_created
ON feedback(user_id, created_at DESC);

-- 2. conversation_id로 빠른 조회
CREATE INDEX idx_feedback_conversation
ON feedback(conversation_id);

-- 3. 대화별 메시지 조회 최적화 (시간 순)
CREATE INDEX idx_feedback_conversation_created
ON feedback(conversation_id, created_at ASC);

-- 4. 첫 메시지 빠른 조회
CREATE INDEX idx_feedback_first_message
ON feedback(conversation_id, is_first_message)
WHERE is_first_message = true;

-- 5. UUID로 빠른 조회 (피드백 업데이트용)
-- UUID는 PRIMARY KEY라서 자동으로 인덱스 생성됨

-- =====================================================
-- Step 4: 테스트 데이터 삽입 (선택 사항)
-- =====================================================

-- 사용자 1의 대화 1 (연말정산 세액공제)
INSERT INTO feedback (
    user_id,
    conversation_id,
    conversation_title,
    is_first_message,
    message,
    response,
    created_at
) VALUES
(
    'test123',
    'conv_test123_1733280000000',
    '연말정산 세액공제 문의',
    true,
    '연말정산이 뭐야?',
    '연말정산은 1년 동안 납부한 세금을 정산하는 절차입니다. 소득공제와 세액공제를 통해 환급받을 수 있습니다.',
    '2025-12-03 10:00:00'
),
(
    'test123',
    'conv_test123_1733280000000',
    NULL,  -- 제목은 첫 메시지에만
    false,
    '공제 항목은?',
    '주요 공제 항목으로는 의료비, 교육비, 기부금, 주택자금 등이 있습니다.',
    '2025-12-03 10:01:00'
),
(
    'test123',
    'conv_test123_1733280000000',
    NULL,
    false,
    '의료비는 얼마까지 공제돼?',
    '의료비는 총급여의 3%를 초과하는 금액에 대해 15%가 세액공제됩니다.',
    '2025-12-03 10:02:00'
);

-- 사용자 1의 대화 2 (부양가족 등록)
INSERT INTO feedback (
    user_id,
    conversation_id,
    conversation_title,
    is_first_message,
    message,
    response,
    created_at
) VALUES
(
    'test123',
    'conv_test123_1733285000000',
    '부양가족 등록 방법',
    true,
    '부양가족 등록하는 방법 알려줘',
    '부양가족은 국세청 홈택스에서 등록할 수 있습니다. 연간 소득이 100만원 이하인 직계존속 및 직계비속이 대상입니다.',
    '2025-12-03 14:00:00'
),
(
    'test123',
    'conv_test123_1733285000000',
    NULL,
    false,
    '나이 제한은?',
    '직계존속은 만 60세 이상, 직계비속은 만 20세 이하입니다.',
    '2025-12-03 14:01:00'
);

-- 사용자 2의 대화 1
INSERT INTO feedback (
    user_id,
    conversation_id,
    conversation_title,
    is_first_message,
    message,
    response,
    created_at
) VALUES
(
    'user456',
    'conv_user456_1733290000000',
    '신용카드 소득공제',
    true,
    '신용카드 소득공제 한도는?',
    '신용카드 소득공제는 총급여의 25%를 초과하는 금액에 대해 15~30% 공제됩니다. 한도는 연간 300만원입니다.',
    '2025-12-03 16:00:00'
);

-- =====================================================
-- Step 5: 검증 쿼리
-- =====================================================

-- 1. 테이블 구조 확인
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'feedback'
ORDER BY ordinal_position;

-- 2. 인덱스 확인
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'feedback'
ORDER BY indexname;

-- 3. 테스트 데이터 확인
SELECT
    COUNT(*) as total_messages,
    COUNT(DISTINCT user_id) as total_users,
    COUNT(DISTINCT conversation_id) as total_conversations
FROM feedback;

-- 4. 대화 목록 조회 테스트 (사용자별)
SELECT
    conversation_id,
    MAX(conversation_title) as title,
    MIN(message) FILTER (WHERE is_first_message = true) as first_message,
    COUNT(*) as message_count,
    MIN(created_at) as created_at,
    MAX(created_at) as updated_at
FROM feedback
WHERE user_id = 'test123'
GROUP BY conversation_id
ORDER BY MAX(created_at) DESC;

-- 5. 특정 대화의 메시지 조회 테스트
SELECT
    CASE
        WHEN message IS NOT NULL THEN 'user'
        ELSE 'assistant'
    END as role,
    COALESCE(message, response) as content,
    created_at
FROM feedback
WHERE conversation_id = 'conv_test123_1733280000000'
ORDER BY created_at ASC;

-- =====================================================
-- Step 6: 통계 정보 업데이트
-- =====================================================

ANALYZE feedback;

-- =====================================================
-- 완료 메시지
-- =====================================================

DO $$
BEGIN
    RAISE NOTICE '=====================================================';
    RAISE NOTICE '데이터베이스 재생성 완료!';
    RAISE NOTICE '=====================================================';
    RAISE NOTICE '테이블: feedback';
    RAISE NOTICE '인덱스: 5개 생성됨';
    RAISE NOTICE '테스트 데이터: % 건', (SELECT COUNT(*) FROM feedback);
    RAISE NOTICE '=====================================================';
END $$;
