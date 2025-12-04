-- =====================================================
-- 데이터베이스 재생성 (UUID 기반 conversation_id)
-- =====================================================
-- 작성일: 2025-12-04
-- 개선: 첫 메시지의 uuid를 conversation_id로 사용
-- 장점: 별도 ID 생성 로직 불필요, 간단함

-- =====================================================
-- Step 1: 기존 테이블 삭제
-- =====================================================

DROP TABLE IF EXISTS feedback CASCADE;

-- =====================================================
-- Step 2: 새 테이블 생성
-- =====================================================

CREATE TABLE feedback (
    -- 기본 필드
    uuid UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(50) NOT NULL,

    -- 대화 관리 (conversation_id는 첫 메시지의 uuid와 동일)
    conversation_id UUID,  -- 첫 메시지의 uuid 사용
    conversation_title TEXT,
    is_first_message BOOLEAN DEFAULT false,

    -- 메시지 내용
    message TEXT,
    response TEXT,

    -- 시간 정보
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- 피드백 정보
    feedback VARCHAR(10),
    reason TEXT,
    comment TEXT,
    name VARCHAR(100),
    timestamp TIMESTAMP,
    feedback_updated_at TIMESTAMP,

    -- 제약조건: conversation_id가 있으면 반드시 존재하는 uuid여야 함
    CONSTRAINT fk_conversation_first_message
        FOREIGN KEY (conversation_id)
        REFERENCES feedback(uuid)
        ON DELETE CASCADE
        DEFERRABLE INITIALLY DEFERRED  -- 첫 메시지 삽입 시 지연 체크
);

-- =====================================================
-- Step 3: 인덱스 생성
-- =====================================================

CREATE INDEX idx_feedback_user_created
ON feedback(user_id, created_at DESC);

CREATE INDEX idx_feedback_conversation
ON feedback(conversation_id);

CREATE INDEX idx_feedback_conversation_created
ON feedback(conversation_id, created_at ASC);

CREATE INDEX idx_feedback_first_message
ON feedback(conversation_id, is_first_message)
WHERE is_first_message = true;

-- =====================================================
-- Step 4: 테스트 데이터 삽입
-- =====================================================

-- 사용자 1의 대화 1 (연말정산 세액공제)
-- 첫 메시지: uuid를 미리 생성
DO $$
DECLARE
    first_msg_uuid UUID := gen_random_uuid();
BEGIN
    -- 첫 메시지
    INSERT INTO feedback (
        uuid,
        user_id,
        conversation_id,
        conversation_title,
        is_first_message,
        message,
        response,
        created_at
    ) VALUES (
        first_msg_uuid,  -- uuid
        'test123',
        first_msg_uuid,  -- conversation_id = uuid (첫 메시지)
        '연말정산 세액공제 문의',
        true,
        '연말정산이 뭐야?',
        '연말정산은 1년 동안 납부한 세금을 정산하는 절차입니다. 소득공제와 세액공제를 통해 환급받을 수 있습니다.',
        '2025-12-03 10:00:00'
    );

    -- 두 번째 메시지 (같은 conversation_id 사용)
    INSERT INTO feedback (
        user_id,
        conversation_id,
        is_first_message,
        message,
        response,
        created_at
    ) VALUES (
        'test123',
        first_msg_uuid,  -- 첫 메시지의 uuid 사용
        false,
        '공제 항목은?',
        '주요 공제 항목으로는 의료비, 교육비, 기부금, 주택자금 등이 있습니다.',
        '2025-12-03 10:01:00'
    );

    -- 세 번째 메시지
    INSERT INTO feedback (
        user_id,
        conversation_id,
        is_first_message,
        message,
        response,
        created_at
    ) VALUES (
        'test123',
        first_msg_uuid,
        false,
        '의료비는 얼마까지 공제돼?',
        '의료비는 총급여의 3%를 초과하는 금액에 대해 15%가 세액공제됩니다.',
        '2025-12-03 10:02:00'
    );
END $$;

-- 사용자 1의 대화 2 (부양가족 등록)
DO $$
DECLARE
    first_msg_uuid UUID := gen_random_uuid();
BEGIN
    INSERT INTO feedback (
        uuid,
        user_id,
        conversation_id,
        conversation_title,
        is_first_message,
        message,
        response,
        created_at
    ) VALUES (
        first_msg_uuid,
        'test123',
        first_msg_uuid,
        '부양가족 등록 방법',
        true,
        '부양가족 등록하는 방법 알려줘',
        '부양가족은 국세청 홈택스에서 등록할 수 있습니다. 연간 소득이 100만원 이하인 직계존속 및 직계비속이 대상입니다.',
        '2025-12-03 14:00:00'
    );

    INSERT INTO feedback (
        user_id,
        conversation_id,
        is_first_message,
        message,
        response,
        created_at
    ) VALUES (
        'test123',
        first_msg_uuid,
        false,
        '나이 제한은?',
        '직계존속은 만 60세 이상, 직계비속은 만 20세 이하입니다.',
        '2025-12-03 14:01:00'
    );
END $$;

-- 사용자 2의 대화 1
DO $$
DECLARE
    first_msg_uuid UUID := gen_random_uuid();
BEGIN
    INSERT INTO feedback (
        uuid,
        user_id,
        conversation_id,
        conversation_title,
        is_first_message,
        message,
        response,
        created_at
    ) VALUES (
        first_msg_uuid,
        'user456',
        first_msg_uuid,
        '신용카드 소득공제',
        true,
        '신용카드 소득공제 한도는?',
        '신용카드 소득공제는 총급여의 25%를 초과하는 금액에 대해 15~30% 공제됩니다. 한도는 연간 300만원입니다.',
        '2025-12-03 16:00:00'
    );
END $$;

-- =====================================================
-- Step 5: 검증 쿼리
-- =====================================================

-- 1. 테이블 구조 확인
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_name = 'feedback'
ORDER BY ordinal_position;

-- 2. 인덱스 확인
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'feedback'
ORDER BY indexname;

-- 3. 테스트 데이터 확인
SELECT
    COUNT(*) as total_messages,
    COUNT(DISTINCT user_id) as total_users,
    COUNT(DISTINCT conversation_id) as total_conversations
FROM feedback;

-- 4. conversation_id = uuid 검증 (첫 메시지만)
SELECT
    uuid,
    conversation_id,
    uuid = conversation_id as is_first_message_check,
    is_first_message,
    message
FROM feedback
WHERE is_first_message = true;

-- 5. 대화 목록 조회 (첫 메시지 기준)
SELECT
    f1.conversation_id,
    f1.conversation_title as title,
    f1.message as first_message,
    (
        SELECT COUNT(*)
        FROM feedback f2
        WHERE f2.conversation_id = f1.conversation_id
    ) as message_count,
    f1.created_at,
    (
        SELECT MAX(created_at)
        FROM feedback f3
        WHERE f3.conversation_id = f1.conversation_id
    ) as updated_at
FROM feedback f1
WHERE f1.user_id = 'test123'
AND f1.is_first_message = true
ORDER BY updated_at DESC;

-- 6. 특정 대화의 메시지 조회
SELECT
    uuid,
    CASE
        WHEN message IS NOT NULL THEN 'user'
        ELSE 'assistant'
    END as role,
    COALESCE(message, response) as content,
    created_at
FROM feedback
WHERE conversation_id = (
    SELECT conversation_id
    FROM feedback
    WHERE user_id = 'test123'
    AND is_first_message = true
    ORDER BY created_at DESC
    LIMIT 1
)
ORDER BY created_at ASC;

-- =====================================================
-- Step 6: 통계 정보 업데이트
-- =====================================================

ANALYZE feedback;

-- =====================================================
-- 완료 메시지
-- =====================================================

DO $$
DECLARE
    msg_count INTEGER;
    conv_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO msg_count FROM feedback;
    SELECT COUNT(DISTINCT conversation_id) INTO conv_count FROM feedback;

    RAISE NOTICE '=====================================================';
    RAISE NOTICE '데이터베이스 재생성 완료! (UUID 기반)';
    RAISE NOTICE '=====================================================';
    RAISE NOTICE '테이블: feedback';
    RAISE NOTICE '인덱스: 5개 생성됨';
    RAISE NOTICE '메시지: % 건', msg_count;
    RAISE NOTICE '대화: % 건', conv_count;
    RAISE NOTICE 'conversation_id = 첫 메시지의 uuid 사용';
    RAISE NOTICE '=====================================================';
END $$;
