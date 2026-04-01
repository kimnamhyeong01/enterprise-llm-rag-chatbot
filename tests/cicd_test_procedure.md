# CI/CD 파이프라인 테스트 절차

## 목차

1. [사전 준비](#1-사전-준비)
2. [Jenkins 자격증명 설정](#2-jenkins-자격증명-설정)
3. [CI 단계별 테스트](#3-ci-단계별-테스트)
4. [CD 단계별 테스트](#4-cd-단계별-테스트)
5. [통합 파이프라인 실행 확인](#5-통합-파이프라인-실행-확인)
6. [롤백 시나리오 테스트](#6-롤백-시나리오-테스트)
7. [문제 해결 (Troubleshooting)](#7-문제-해결-troubleshooting)

---

## 1. 사전 준비

### 1-1. 인프라 구성 확인

| 항목 | 값 | 확인 방법 |
|------|----|-----------|
| Jenkins URL | `http://localhost:8080` | 브라우저 접속 |
| Harbor 레지스트리 | `amdp-registry.skala-ai.com` | `docker login` 시도 |
| 배포 대상 서버 | `DEPLOY_HOST` 환경변수로 지정 | SSH 접속 확인 |
| 배포 디렉토리 | `/opt/rag-chatbot` | 서버에서 `ls` 확인 |

### 1-2. Jenkins 파이프라인 환경변수 (`DEPLOY_HOST`)

Jenkins > 해당 파이프라인 > **Configure** > **This project is parameterized** 에서 아래 파라미터를 추가한다.

| 파라미터명 | 타입 | 예시값 |
|------------|------|--------|
| `DEPLOY_HOST` | String | `192.168.1.100` |

또는 Jenkins 전역 환경변수 설정:
**Manage Jenkins > Configure System > Global properties > Environment variables**

---

## 2. Jenkins 자격증명 설정

### 2-1. Harbor 레지스트리 자격증명

**Manage Jenkins > Credentials > System > Global credentials > Add Credentials**

| 필드 | 값 |
|------|----|
| Kind | Username with password |
| ID | `harbor-cred` |
| Username | Harbor 계정 ID |
| Password | Harbor 계정 패스워드 |

### 2-2. 배포 서버 SSH 자격증명

| 필드 | 값 |
|------|----|
| Kind | SSH Username with private key |
| ID | `deploy-server-ssh` |
| Username | `deploy` |
| Private Key | 배포 서버 접근용 개인키 (PEM 형식) |

**사전 확인**: 수동으로 SSH 접속이 되는지 확인

```bash
ssh -i /path/to/private.key deploy@<DEPLOY_HOST>
```

### 2-3. 배포 서버 환경변수 설정

배포 서버 `/opt/rag-chatbot/.env` 파일에 아래 내용을 설정한다.

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
HARBOR_USER=<harbor-username>
HARBOR_PASS=<harbor-password>
```

---

## 3. CI 단계별 테스트

### 3-1. build-backend 단계

**목적**: 백엔드 Docker 이미지가 빌드오류 없이 생성되는지 확인

**검증 방법**:
```bash
# 로컬에서 직접 확인
docker build -t my-backend:test ./backend
docker images | grep my-backend
```

**기대 결과**:
- `my-backend:<BUILD_NUMBER>` 이미지가 생성됨
- Jenkins 콘솔에서 `Successfully built ...` 메시지 확인

---

### 3-2. build-frontend 단계

**목적**: 프론트엔드 Docker 이미지가 빌드오류 없이 생성되는지 확인

**검증 방법**:
```bash
docker build -t my-frontend:test ./frontend
docker images | grep my-frontend
```

**기대 결과**:
- `my-frontend:<BUILD_NUMBER>` 이미지가 생성됨

---

### 3-3. test 단계 (스모크 테스트)

**목적**: 백엔드 컨테이너 내부에서 Python 모듈 임포트가 정상 동작하는지 확인

**검증 방법**:
```bash
# 로컬에서 수동 실행
docker run --rm \
  -e OPENAI_API_KEY=dummy \
  my-backend:test \
  python -c "from main import app; print('Import OK')"
```

**기대 결과**:
- `Import OK` 출력 후 컨테이너 종료 (exit 0)

---

### 3-4. push 단계

**목적**: 빌드된 이미지가 Harbor 레지스트리에 정상적으로 push되는지 확인

**검증 방법**:

1. Jenkins 콘솔에서 `docker push` 완료 로그 확인
2. Harbor UI에서 이미지 존재 여부 확인:
   - `amdp-registry.skala-ai.com/skala26a-ai2/my-backend:<BUILD_NUMBER>`
   - `amdp-registry.skala-ai.com/skala26a-ai2/my-frontend:<BUILD_NUMBER>`
3. 또는 CLI로 확인:
   ```bash
   docker pull amdp-registry.skala-ai.com/skala26a-ai2/my-backend:<BUILD_NUMBER>
   ```

**기대 결과**:
- `BUILD_NUMBER` 태그와 `latest` 태그 모두 push됨
- Harbor UI에서 2개 태그 확인 가능

---

## 4. CD 단계별 테스트

### 4-1. deploy 단계

**목적**: 배포 서버에서 컨테이너가 최신 이미지로 재시작되는지 확인

**검증 방법** (배포 서버에서 직접):
```bash
# Harbor에서 이미지가 pull 되었는지 확인
docker images | grep my-backend
docker images | grep my-frontend

# 컨테이너가 실행 중인지 확인
docker compose -f /opt/rag-chatbot/docker-compose.deploy.yml ps
```

**기대 결과**:
```
NAME            IMAGE                                           STATUS
rag-backend     amdp-registry.../my-backend:<BUILD_NUMBER>     Up (healthy)
rag-frontend    amdp-registry.../my-frontend:<BUILD_NUMBER>    Up
```

---

### 4-2. verify 단계

**목적**: 배포된 서비스가 실제로 응답하는지 자동 확인

**Jenkins가 수행하는 검증**:
1. 배포 후 20초 대기 (서비스 기동 시간)
2. 백엔드 헬스 체크 엔드포인트 호출 (최대 5회 재시도)
3. 프론트엔드 HTTP 200 응답 확인

**수동 검증**:
```bash
# 배포 서버에서 실행
curl http://localhost:8000/health
# 기대: {"status": "ok"}

curl -o /dev/null -w "%{http_code}" http://localhost:8501
# 기대: 200
```

---

## 5. 통합 파이프라인 실행 확인

### 5-1. 파이프라인 전체 실행 절차

1. Jenkins 대시보드 접속 (`http://localhost:8080`)
2. 해당 파이프라인 선택
3. **Build with Parameters** 클릭
4. `DEPLOY_HOST` 값 입력 후 **Build** 클릭
5. **Console Output** 에서 각 스테이지 진행 확인

### 5-2. 파이프라인 스테이지별 성공 기준

| 스테이지 | 성공 기준 |
|----------|-----------|
| build-backend | 이미지 빌드 완료 (exit 0) |
| build-frontend | 이미지 빌드 완료 (exit 0) |
| test | `Import OK` 출력 (exit 0) |
| push | Harbor에 이미지 push 완료 |
| deploy | 배포 서버 컨테이너 재시작 완료 |
| verify | 백엔드 `/health` 200 응답, 프론트엔드 200 응답 |

### 5-3. 파이프라인 Stage View 확인

Jenkins Blue Ocean 또는 Stage View 플러그인에서 아래와 같이 모든 스테이지가 녹색(성공)인지 확인한다.

```
build-backend → build-frontend → test → push → deploy → verify
     ✅               ✅           ✅      ✅       ✅        ✅
```

---

## 6. 롤백 시나리오 테스트

### 6-1. 롤백 트리거 조건

- `deploy` 또는 `verify` 스테이지 실패 시 `post { failure { ... } }` 블록이 자동 실행
- `latest` 태그 이미지로 롤백

### 6-2. 롤백 수동 테스트 방법

배포 서버에서 직접 이전 버전으로 롤백:
```bash
cd /opt/rag-chatbot

# 특정 빌드 번호로 롤백
PREV_BUILD=<이전_빌드_번호>

# docker-compose 파일의 이미지 태그 변경
sed -i "s/:<현재_빌드번호>/:${PREV_BUILD}/g" docker-compose.deploy.yml

# 재시작
docker compose -f docker-compose.deploy.yml up -d

# 헬스 체크
curl http://localhost:8000/health
```

### 6-3. 롤백 성공 기준

- 이전 버전 컨테이너가 정상 기동
- `/health` 엔드포인트 200 응답 복구

---

## 7. 문제 해결 (Troubleshooting)

### 7-1. `harbor-cred` 인증 실패

**증상**: `unauthorized: authentication required`

**해결**:
1. Jenkins Credentials에서 ID가 정확히 `harbor-cred`인지 확인
2. Harbor 계정 비밀번호 만료 여부 확인
3. 수동으로 `docker login` 시도하여 자격증명 검증

```bash
docker login amdp-registry.skala-ai.com -u <user>
```

---

### 7-2. SSH 연결 실패 (`deploy-server-ssh`)

**증상**: `Permission denied (publickey)` 또는 `Connection refused`

**해결**:
1. 배포 서버의 `~/.ssh/authorized_keys`에 공개키 등록 여부 확인
2. Jenkins Credentials에 등록된 개인키가 배포 서버 공개키와 쌍이 맞는지 확인
3. `DEPLOY_HOST` 환경변수가 올바르게 설정되었는지 확인

```bash
# Jenkins 서버에서 수동 테스트
ssh -o StrictHostKeyChecking=no deploy@<DEPLOY_HOST> 'echo OK'
```

---

### 7-3. verify 단계 헬스 체크 실패

**증상**: `curl: (7) Failed to connect` 또는 타임아웃

**해결**:
1. 배포 서버에서 컨테이너 상태 확인
   ```bash
   docker compose -f /opt/rag-chatbot/docker-compose.deploy.yml ps
   docker logs rag-backend --tail 50
   ```
2. `OPENAI_API_KEY`가 배포 서버 `.env`에 설정되어 있는지 확인
3. 포트 방화벽 허용 여부 확인 (8000, 8501)

---

### 7-4. 이미지 push 실패 (용량/권한)

**증상**: `denied: requested access to the resource is denied`

**해결**:
1. Harbor 프로젝트 `skala26a-ai2`에 해당 계정의 **Push** 권한 확인
2. Harbor 프로젝트 스토리지 할당량 확인

---

### 7-5. 로컬 임시 이미지 정리 실패

**증상**: `post { always }` 블록에서 `docker rmi` 오류

**영향**: 파이프라인 결과에는 영향 없음 (`|| true` 처리됨)

**해결**: Jenkins 서버에서 주기적으로 이미지 정리
```bash
docker image prune -f
```
