# Agent Entry

`dyna_rl`은 미분 가능한 simulator와 학습 dynamics를 이용한 정책 학습 연구 저장소다.
사용자 규칙은 `~/.codex/AGENTS.md`, 역할과 실행 명령은 [README](README.md), 환경 경계는
[환경 계약](docs/environment.md)을 따른다.

## 규칙

1. 실행은 `$HOME/.venvs/rewarped`의 PyTorch·Rewarped를 사용한다. Package 추가·버전 변경은 사용자 승인 뒤에만 하고
   `uv sync --dry-run`으로 변경 범위를 먼저 확인한다. 다른 저장소의 환경은 변경하지 않는다.
2. 구현 source 편집은 명시적으로 승인된 범위에서만 한다. 사용자가 학습하며 작성하는 알고리즘을 요청 없이 완성본으로
   대신 채우지 않는다. 문서와 검증은 작업 범위 안에서 진행한다.
3. 루트 폴더는 `media/`, `docs/`, `tests/`, `source/`, `scripts/`다. 알고리즘은 `source/wm_skills/`에서
   `BPTT.py`·`SHAC.py`·`SAPO.py`처럼 한 파일씩 작성하고, 실행 조립은 `scripts/`에 둔다.
4. Task는 `source/<task>/`, 로봇 자산과 license는 `source/robot/<robot>/`에 둔다. 필요가 확인되기 전에는
   backend dispatcher·공통 base class·빈 알고리즘 파일을 추가하지 않는다.
5. 검증한 범용 구성요소는 `/home/frlab/skills/world_model_skills`로 이관한다. 대상 library의 계약을 먼저 읽고,
   simulator·consumer import 없이 분리한다. 이관 뒤 이 저장소는 library를 import한다.
6. 학습 run의 시작·중단·재개는 사용자가 정한다. 환경 설치 검사와 학습 run을 구분한다.
7. 주석과 docstring은 마침표 없는 짧은 한국어 명사형으로 쓴다. 기계적 정렬은 `ruff check --fix`·`ruff format`으로
   하고 diff를 확인한다.
8. 커밋은 `main`에서 경로를 명시해 stage하고, push는 요청받을 때만 한다.
