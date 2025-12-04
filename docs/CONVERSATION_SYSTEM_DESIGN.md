# 대화 히스토리 관리 시스템 설계

ChatGPT 스타일의 사용자별 대화 목록 관리 및 이어하기 기능

## 📋 목차
1. [개요](#개요)
2. [현재 시스템 분석](#현재-시스템-분석)
3. [데이터베이스 설계](#데이터베이스-설계)
4. [API 설계](#api-설계)
5. [프론트엔드 UI 설계](#프론트엔드-ui-설계)
6. [구현 단계](#구현-단계)

---

## 개요

### 목표
- 사용자별 대화 목록을 관리
- 이전 대화를 다시 열어서 이어하기 가능
- 대화 제목 자동 생성 및 수정
- 대화 삭제 및 검색 기능

### 주요 기능
- ✅ 새 대화 시작
- ✅ 대화 목록 조회 (사용자별)
- ✅ 특정 대화 선택 및 히스토리 로드
- ✅ 대화 제목 자동 생성
- ✅ 대화 제목 수정
- ✅ 대화 삭제
- ✅ 대화 검색 (선택 사항)

---

## 현재 시스템 분석

### 현재 구조
```
1. session_id (클라이언트 생성)
   - 형식: "userId_timestamp_random"
   - 예: "test123_1733285240000_a3f2c1"

2. feedback 테이블
   - uuid: 메시지 고유 ID
   - user_id: 사용자 ID
   - message: 사용자 질문
   - response: AI 응답
   - created_at: 생성 시간
   - feedback, reason, comment: 피드백 정보

3. 대화 히스토리 (LangChain)
   - InMemoryChatMessageHistory
   - session_id 기반으로 메모리에 저장
   - 서버 재시작 시 초기화됨
```

### 문제점
- ❌ 대화 목록이 없음 (사용자가 이전 대화를 찾을 수 없음)
- ❌ 대화 제목이 없음 (무엇에 대한 대화인지 알 수 없음)
- ❌ 새 대화/기존 대화 구분 없음
- ❌ 서버 재시작 시 대화 히스토리 소실 (InMemory 사용)

---

## 데이터베이스 설계

### Option 1: 기존 feedback 테이블 활용 (권장)

**장점**: 기존 데이터 유지, 마이그레이션 불필요
**단점**: conversation 개념이 명확하지 않음

#### 1-1. feedback 테이블에 conversation_id 추가

```sql
-- feedback 테이블 수정
ALTER TABLE feedback
ADD COLUMN conversation_id VARCHAR(100),
ADD COLUMN conversation_title TEXT,
ADD COLUMN is_first_message BOOLEAN DEFAULT false;

-- 인덱스 추가 (성능 최적화)
CREATE INDEX idx_feedback_conversation ON feedback(conversation_id);
CREATE INDEX idx_feedback_user_created ON feedback(user_id, created_at DESC);
```

**테이블 구조**:
```sql
feedback:
  - uuid (PK)
  - user_id
  - conversation_id       ← NEW (대화 그룹 ID)
  - conversation_title    ← NEW (대화 제목, 첫 메시지에만 저장)
  - is_first_message      ← NEW (대화 제목 생성용)
  - message
  - response
  - created_at
  - feedback, reason, comment, name
  - feedback_updated_at, timestamp
```

**데이터 예시**:
```
conversation_id: "conv_test123_1733285240000"
conversation_title: "연말정산 세액공제 문의"

| uuid | user_id | conversation_id | conversation_title | is_first_message | message | response |
|------|---------|-----------------|-------------------|------------------|---------|----------|
| abc1 | test123 | conv_123_001    | 연말정산 세액공제   | true             | 연말정산이 뭐야? | 연말정산은... |
| abc2 | test123 | conv_123_001    | null              | false            | 공제 항목은? | 의료비, 교육비... |
| abc3 | test123 | conv_123_002    | 부양가족 등록      | true             | 부양가족 등록? | 부양가족은... |
```

---

### Option 2: 새 테이블 생성 (확장성 높음)

**장점**: 명확한 구조, 확장 용이
**단점**: 마이그레이션 필요, 복잡도 증가

#### 2-1. conversations 테이블 (대화 메타데이터)

```sql
CREATE TABLE conversations (
    conversation_id VARCHAR(100) PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    title TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    first_message TEXT,
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX idx_conversations_user ON conversations(user_id, updated_at DESC);
```

#### 2-2. messages 테이블 (개별 메시지)

```sql
CREATE TABLE messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_conversation FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

-- 인덱스
CREATE INDEX idx_messages_conversation ON messages(conversation_id, created_at ASC);
```

#### 2-3. 기존 feedback 테이블과 연결

```sql
-- feedback 테이블에 conversation_id 추가
ALTER TABLE feedback
ADD COLUMN conversation_id VARCHAR(100),
ADD CONSTRAINT fk_feedback_conversation
    FOREIGN KEY (conversation_id)
    REFERENCES conversations(conversation_id)
    ON DELETE SET NULL;
```

---

## API 설계

### 1. 대화 목록 조회
```http
GET /conversations?userId={userId}&limit={limit}&offset={offset}
```

**응답**:
```json
{
  "conversations": [
    {
      "conversation_id": "conv_test123_1733285240000",
      "title": "연말정산 세액공제 문의",
      "preview": "연말정산이 뭐야?",
      "message_count": 5,
      "created_at": "2025-12-03T10:00:00",
      "updated_at": "2025-12-03T10:15:00"
    },
    {
      "conversation_id": "conv_test123_1733280000000",
      "title": "부양가족 등록 방법",
      "preview": "부양가족 등록하는 방법 알려줘",
      "message_count": 3,
      "created_at": "2025-12-02T15:30:00",
      "updated_at": "2025-12-02T15:45:00"
    }
  ],
  "total": 25,
  "limit": 20,
  "offset": 0
}
```

---

### 2. 새 대화 시작
```http
POST /conversations
Content-Type: application/json

{
  "userId": "test123"
}
```

**응답**:
```json
{
  "conversation_id": "conv_test123_1733285240000",
  "user_id": "test123",
  "created_at": "2025-12-03T10:00:00"
}
```

---

### 3. 특정 대화의 메시지 조회
```http
GET /conversations/{conversation_id}/messages?limit={limit}
```

**응답**:
```json
{
  "conversation_id": "conv_test123_1733285240000",
  "title": "연말정산 세액공제 문의",
  "messages": [
    {
      "role": "user",
      "content": "연말정산이 뭐야?",
      "created_at": "2025-12-03T10:00:00"
    },
    {
      "role": "assistant",
      "content": "연말정산은 1년 동안 납부한 세금을...",
      "created_at": "2025-12-03T10:00:05"
    },
    {
      "role": "user",
      "content": "공제 항목은?",
      "created_at": "2025-12-03T10:01:00"
    },
    {
      "role": "assistant",
      "content": "의료비, 교육비, 기부금 등이 있습니다...",
      "created_at": "2025-12-03T10:01:03"
    }
  ],
  "message_count": 4,
  "created_at": "2025-12-03T10:00:00",
  "updated_at": "2025-12-03T10:01:03"
}
```

---

### 4. 대화 제목 수정
```http
PATCH /conversations/{conversation_id}
Content-Type: application/json

{
  "title": "연말정산 완전 정복"
}
```

**응답**:
```json
{
  "conversation_id": "conv_test123_1733285240000",
  "title": "연말정산 완전 정복",
  "updated_at": "2025-12-03T10:30:00"
}
```

---

### 5. 대화 삭제
```http
DELETE /conversations/{conversation_id}
```

**응답**:
```json
{
  "status": "deleted",
  "conversation_id": "conv_test123_1733285240000"
}
```

---

### 6. 대화 제목 자동 생성 (내부 로직)

첫 메시지 전송 시 LLM으로 제목 생성:
```python
def generate_conversation_title(first_message: str) -> str:
    """
    첫 메시지를 기반으로 대화 제목 생성 (10자 이내)

    예시:
    - "연말정산이 뭐야?" → "연말정산 문의"
    - "부양가족 등록하는 방법 알려줘" → "부양가족 등록 방법"
    - "의료비 공제 한도는?" → "의료비 공제 한도"
    """
    # LLM 프롬프트
    prompt = f"""
    다음 질문을 10자 이내의 간결한 제목으로 요약하세요.

    질문: {first_message}

    제목 (10자 이내):
    """

    # LLM 호출 (간단한 요약)
    title = llm.invoke(prompt).strip()

    # 길이 제한
    if len(title) > 15:
        title = title[:15] + "..."

    return title
```

---

## 프론트엔드 UI 설계

### 레이아웃 구조

```
┌─────────────────────────────────────────────────┐
│  [Logo]  연말정산 전문 챗봇         [userId]    │
├──────────┬──────────────────────────────────────┤
│          │                                      │
│ 📝 새 대화 │         대화 영역                    │
│          │                                      │
│ 대화 목록  │  안녕하세요! 😊                      │
│          │  연말정산 전문 상담 챗봇입니다.        │
│ 📄 연말정산│                                      │
│   세액공제 │  [사용자 입력창]                      │
│          │  [전송 버튼]                         │
│ 📄 부양가족│                                      │
│   등록 방법│                                      │
│          │                                      │
│ 📄 의료비 │                                      │
│   공제 한도│                                      │
│          │                                      │
└──────────┴──────────────────────────────────────┘
```

### HTML 구조 (chat.html 수정)

```html
<div class="container">
  <!-- 왼쪽 사이드바 (대화 목록) -->
  <div class="sidebar">
    <button id="new-conversation-btn" class="new-conversation-btn">
      ➕ 새 대화
    </button>

    <div class="conversation-list" id="conversation-list">
      <!-- 대화 목록이 여기에 동적으로 추가됨 -->
    </div>
  </div>

  <!-- 오른쪽 채팅 영역 (기존) -->
  <div class="chat-area">
    <div class="chat-messages" id="chat-messages">
      <!-- 메시지들 -->
    </div>

    <form id="chat-form">
      <textarea id="user-input"></textarea>
      <button type="submit">전송</button>
    </form>
  </div>
</div>
```

### CSS (사이드바 추가)

```css
.container {
  display: flex;
  height: 100vh;
}

.sidebar {
  width: 280px;
  background: #f5f5f5;
  border-right: 1px solid #ddd;
  padding: 20px;
  overflow-y: auto;
}

.new-conversation-btn {
  width: 100%;
  padding: 12px;
  background: #4285f4;
  color: white;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 20px;
}

.conversation-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.conversation-item {
  padding: 12px;
  background: white;
  border: 1px solid #ddd;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.conversation-item:hover {
  background: #e3f2fd;
}

.conversation-item.active {
  background: #bbdefb;
  border-color: #4285f4;
}

.conversation-title {
  font-weight: 500;
  margin-bottom: 4px;
}

.conversation-preview {
  font-size: 12px;
  color: #666;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
}
```

### JavaScript 로직

```javascript
// 현재 대화 ID
let currentConversationId = null;

// 페이지 로드 시 대화 목록 조회
async function loadConversationList() {
  const response = await fetch(`/conversations?userId=${userId}`);
  const data = await response.json();

  const listEl = document.getElementById('conversation-list');
  listEl.innerHTML = '';

  data.conversations.forEach(conv => {
    const item = document.createElement('div');
    item.className = 'conversation-item';
    if (conv.conversation_id === currentConversationId) {
      item.classList.add('active');
    }

    item.innerHTML = `
      <div class="conversation-title">${conv.title || '새 대화'}</div>
      <div class="conversation-preview">${conv.preview}</div>
    `;

    item.onclick = () => loadConversation(conv.conversation_id);
    listEl.appendChild(item);
  });
}

// 특정 대화 로드
async function loadConversation(conversationId) {
  currentConversationId = conversationId;

  // 대화 메시지 조회
  const response = await fetch(`/conversations/${conversationId}/messages`);
  const data = await response.json();

  // 채팅 영역에 메시지 표시
  const messagesEl = document.getElementById('chat-messages');
  messagesEl.innerHTML = '';

  data.messages.forEach(msg => {
    addMessage(msg.role, msg.content);
  });

  // 대화 목록 업데이트 (active 표시)
  loadConversationList();
}

// 새 대화 시작
async function startNewConversation() {
  const response = await fetch('/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ userId })
  });

  const data = await response.json();
  currentConversationId = data.conversation_id;

  // 채팅 영역 초기화
  document.getElementById('chat-messages').innerHTML = '';

  // 대화 목록 새로고침
  loadConversationList();
}

// 새 대화 버튼 클릭
document.getElementById('new-conversation-btn').onclick = startNewConversation;

// 메시지 전송 시 conversation_id 포함
async function sendMessage(message) {
  const formData = new FormData();
  formData.append('message', message);
  formData.append('session_id', currentConversationId);  // conversation_id 사용
  formData.append('user_id', userId);

  const response = await fetch('/chat', {
    method: 'POST',
    body: formData
  });

  // 스트리밍 응답 처리...

  // 대화 목록 업데이트 (updated_at 갱신)
  loadConversationList();
}
```

---

## 구현 단계

### Phase 1: 데이터베이스 수정 (30분)
1. ✅ feedback 테이블에 conversation_id, conversation_title 컬럼 추가
2. ✅ 인덱스 추가
3. ✅ 마이그레이션 스크립트 작성

### Phase 2: 백엔드 API 구현 (2시간)
1. ✅ GET /conversations - 대화 목록 조회
2. ✅ POST /conversations - 새 대화 시작
3. ✅ GET /conversations/{id}/messages - 메시지 조회
4. ✅ PATCH /conversations/{id} - 제목 수정
5. ✅ DELETE /conversations/{id} - 대화 삭제
6. ✅ 대화 제목 자동 생성 로직

### Phase 3: 기존 코드 수정 (1시간)
1. ✅ /chat 엔드포인트에서 conversation_id 사용
2. ✅ 첫 메시지 시 제목 자동 생성
3. ✅ LangChain 히스토리를 DB 기반으로 변경

### Phase 4: 프론트엔드 UI (2시간)
1. ✅ 사이드바 추가 (대화 목록)
2. ✅ 새 대화 버튼
3. ✅ 대화 선택 기능
4. ✅ 대화 목록 자동 갱신

### Phase 5: 테스트 및 최적화 (1시간)
1. ✅ 기능 테스트
2. ✅ 성능 최적화 (인덱스, 캐싱)
3. ✅ 에러 처리

---

## 주요 변경 사항 요약

### 1. conversation_id 도입
- **기존**: `session_id = "userId_timestamp_random"`
- **변경**: `conversation_id = "conv_userId_timestamp"`
- **session_id → conversation_id 매핑 유지**

### 2. 대화 제목 자동 생성
- 첫 메시지 전송 시 LLM으로 제목 생성
- 10~15자 이내 간결한 제목

### 3. 대화 히스토리 지속성
- **기존**: InMemoryChatMessageHistory (서버 재시작 시 소실)
- **변경**: DB 기반 히스토리 (영구 저장)

### 4. UI 변경
- **기존**: 단일 채팅 화면
- **변경**: 사이드바 + 채팅 영역 (ChatGPT 스타일)

---

## 예상 결과

### 사용자 경험
1. ✅ 챗봇 접속 시 이전 대화 목록 표시
2. ✅ 새 대화 시작 버튼 클릭 → 빈 채팅 화면
3. ✅ 첫 메시지 전송 → 자동으로 제목 생성
4. ✅ 대화 목록에서 이전 대화 선택 → 히스토리 로드
5. ✅ 대화 이어하기 가능

### 데이터 흐름
```
사용자 접속
  ↓
GET /conversations?userId=test123
  ↓
[대화 목록 표시]
  ↓
사용자가 "새 대화" 클릭
  ↓
POST /conversations { userId: "test123" }
  ↓
conversation_id = "conv_test123_1733285240000" 생성
  ↓
사용자가 첫 메시지 "연말정산이 뭐야?" 전송
  ↓
POST /chat { message, conversation_id, user_id }
  ↓
LLM으로 제목 생성: "연말정산 문의"
  ↓
feedback 테이블에 저장:
  - conversation_id: conv_test123_1733285240000
  - conversation_title: "연말정산 문의"
  - is_first_message: true
  ↓
대화 목록 갱신 (새 대화 추가됨)
```

---

## 추가 기능 (선택 사항)

### 1. 대화 검색
```http
GET /conversations/search?userId={userId}&query={query}
```

### 2. 대화 아카이브
- 삭제 대신 아카이브 (숨김)
- 나중에 복원 가능

### 3. 대화 공유
- 공유 링크 생성
- 읽기 전용 모드

### 4. 대화 내보내기
- PDF, TXT 형식으로 다운로드

---

## 참고 자료

- ChatGPT UI: https://chat.openai.com
- LangChain PostgresChatMessageHistory: https://python.langchain.com/docs/integrations/memory/postgres_chat_message_history
- FastAPI CRUD 패턴: https://fastapi.tiangolo.com/tutorial/sql-databases/
