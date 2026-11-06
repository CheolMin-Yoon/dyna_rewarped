# dyna_rl

미분 가능한 simulator와 학습한 dynamics 모델을 이용하는 정책 학습 연구 저장소다.
BPTT·SHAC·SAPO·RFO를 구현·검증하고, 범용 구성요소를 `skills/world_model_skills`로 이관한다.
MJLab 기반 SuperMimic과 LocoWM·RFO 같은 학습에 필요한 구성요소의 개발·검증을 담당한다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `media/` | 그림·동영상 |
| `docs/` | 수식·실험 설계·측정·task 계약 |
| `tests/` | 환경 연결·gradient·알고리즘 검증 |
| `source/wm_skills/` | `BPTT.py`·`SHAC.py`·`SAPO.py`처럼 알고리즘별 한 파일 |
| `source/<task>/` | 관측·action·보상·종료·reference와 simulator 연결 |
| `source/robot/<robot>/` | 로봇 자산과 license |
| `scripts/` | 학습·평가·재생 실행 진입점 |

알고리즘은 한 파일에서 연결해 검증한 뒤, 독립적으로 재사용할 부분을 library로 분해한다.
Simulator·로봇·task 연결은 이 저장소에 두고, 이관한 구현은 library에서 import한다.
`source/wm_skills/BPTT.py`는 JAX로 작성한 수식 참고용 원문이며, PyTorch·Rewarped 실행 대상과 구분한다.

## 환경과 검증

Python과 package 버전은 [pyproject.toml](pyproject.toml)과 `uv.lock`으로 고정한다.
실행 환경은 `$HOME/.venvs/rewarped`이며 GPU가 필요하다.

```bash
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/rewarped" uv sync --locked
source "$HOME/.venvs/rewarped/bin/activate"
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

환경 검증은 렌더링 없이 Rewarped Ant 2개를 2 step 전진시키고 보상에서 action으로의 역전파를 확인한다.
학습 실행은 `scripts/`의 진입점을 작성한 뒤 사용자가 정한다.
Rewarped의 버전 경계와 진단 메시지는 [환경 계약](docs/environment.md)을 따른다.
