# AI Arena

AI(Claude Code, Codex CLI)끼리 게임으로 겨루게 하고, 그 과정을 실시간으로 관전하는 실험 프로젝트입니다.
최종 목표는 사용자가 자기 AI를 훈련하거나 실시간으로 지시하고, 그 경기를 지켜보는 게임입니다.

- 모든 구성요소는 **로컬(127.0.0.1)** 에서만 동작합니다. 서버 배포는 없습니다.
- AI는 **구독으로 로그인된 CLI**(`claude -p`, `codex exec`)로만 호출합니다. API 키 과금은 구조적으로 막혀 있습니다 (아래 [과금 제한](#과금-제한)).

## 지금까지의 실험

| 단계 | 게임 | AI의 역할 | 상태 | 문서 |
|---|---|---|---|---|
| 1 | 9×9 오목 | 매 턴 직접 착수 | 완료 | [AI_Arena_PRD.md](AI_Arena_PRD.md) |
| 2 | Coach Pong (실시간 탁구) | **코치**: 득점 사이 작전 타임에 봇의 작전을 정함 | 완료 | [AI_Arena_Pong_PRD.md](AI_Arena_Pong_PRD.md) |
| 3 | Coach Tanks (diep.io 스타일 탱크) | **코치**: 빌드·클래스·행동 방침을 정하고, 사건이 생길 때마다 수정 | 완료 | [AI_Arena_Tanks_PRD.md](AI_Arena_Tanks_PRD.md) |
| 4 | 3D 게임 | 설계 예정 | — | — |

### Claude 대 Codex 전적

| 게임 | 승자 | 결과 | 기록 |
|---|---|---|---|
| 오목 | Codex | 16수 만에 세로 5목 | [로그](matches/20261122-031432-84bcaf.jsonl) |
| Coach Pong | Codex | 7 : 3 | [로그](matches/pong/20261122-034546-0d649f.jsonl) |
| Coach Tanks | Claude | 3킬 0데스, 레벨 30 : 15 | [로그](matches/tanks/20261122-044141-cb2647.jsonl) |

모든 경기 목록은 [matches/README.md](matches/README.md)에 있습니다.

### 배운 점

- **AI는 실시간 조작을 직접 할 수 없습니다.** CLI 응답에 2~30초가 걸려서, 실시간 게임에서는 "AI = 코치, 봇 = 실행" 구조가 필요했습니다.
- **작전 타임에는 경기를 멈춥니다.** 응답이 빠른 AI가 구조적으로 유리해지지 않게 하기 위해서입니다. 두 코치에게는 항상 동시에 묻습니다.
- **코치는 자기가 관찰한 것만 압니다.** 상대 작전은 숨기고, 관찰 가능한 통계만 브리핑으로 줍니다.
- **결정적(deterministic) 시뮬레이션 덕분에 리플레이로 검증할 수 있습니다.** 시드와 작전 기록만으로 경기 전체를 재현해 결과를 검증합니다.
- **균형 조정은 대량 시뮬레이션으로 했습니다.** 더미 코치끼리 수백 판을 돌려 지배적인 작전이 없는지 확인했고, 그 과정에서 총알 중복 피해 같은 버그도 찾았습니다.
- 경기에서 진 AI는 대개 **판단을 잘못 바꿨습니다.** 탁구의 Claude는 체력 걱정에 이동 속도를 낮췄고, 탱크의 Codex는 체력을 0으로 둔 빌드에 처치당한 뒤 지나치게 수비적으로 바뀌었습니다.

## 구조

```text
arena/
├── agents/      # 공통 에이전트 인터페이스, 더미 에이전트, Claude/Codex CLI 어댑터, 과금 점검
├── coaching.py  # 게임 공통 코칭: 코치 동시 호출, 작전 적용, 이벤트 기록
├── engine/      # 오목 규칙
├── runner/      # 오목 턴 진행
├── pong/        # 탁구 물리, 작전, 코치, 경기 진행, 리플레이
├── tanks/       # 탱크 규칙·엔진·봇·코치·경기 진행·리플레이
├── events/      # 이벤트 로그(JSONL)와 리플레이 검증
├── web/         # 로컬 관전 서버(FastAPI + WebSocket)와 관전 화면(HTML + Canvas)
└── records.py   # 경기 기록 목록 생성
matches/         # 종료된 경기 로그 (진행 중·중단된 경기는 .part로 남고 git에서 제외)
tests/           # pytest (실제 AI를 호출하지 않음)
```

## 실행

[uv](https://docs.astral.sh/uv/)와 Python 3.12 이상이 필요합니다.

```bash
uv sync
uv run pytest                     # 테스트 (실제 AI 호출 없음)

# 더미끼리 (사용량 없음)
uv run python -m arena serve --black heuristic --white random --dummy-delay 0.3
uv run python -m arena pong --left random --right fixed
uv run python -m arena tanks --first random --second fixed

# Claude 대 Codex (구독 사용량 소모, 과금 사전 점검 통과 필요)
uv run python -m arena serve --black claude --white codex
uv run python -m arena pong --left claude --right codex
uv run python -m arena tanks --first claude --second codex

# 기록
uv run python -m arena replay matches/tanks/<id>.jsonl   # 리플레이 검증
uv run python -m arena records                           # matches/README.md 다시 만들기
```

관전 화면은 `http://127.0.0.1:8000` 에서 자동으로 열립니다. 터미널에서만 빠르게 돌리려면 `--fast`를 붙입니다(탁구, 탱크).

## 과금 제한

이 프로젝트는 어떤 경우에도 구독 사용량 외의 토큰을 쓰지 않습니다.

- AI 호출은 `claude -p`와 `codex exec` CLI로만 합니다. 모델 SDK는 의존성에 없습니다.
- AI 프로세스에는 허용 목록에 있는 환경변수(`PATH`, `HOME` 등)만 넘깁니다. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` 같은 키는 전달되지 않습니다.
- 실제 AI 경기는 시작 전에 `claude auth status`(claude.ai 로그인)와 `codex login status`(ChatGPT 로그인)를 확인하고, 아니면 시작하지 않습니다.
- Claude는 `--restricted`로 사용자 설정(`apiKeyHelper` 등)을 무시하고, Codex는 `--ignore-user-config`로 제공자 재정의를 무시합니다.
- 구독 한도를 넘었을 때의 추가 과금(extra usage, credits)은 코드로 막을 수 없으므로 계정 설정에서 꺼 두어야 합니다.
