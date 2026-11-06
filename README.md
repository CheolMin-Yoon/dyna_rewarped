# dyna_rewarped

Rewarped에서 `skills/world_model_skills`의 정책 학습 알고리즘을 검증하는 consumer다.
알고리즘과 범용 모델은 library가 소유하고, 이 저장소는 task·rollout·학습 조립·checkpoint·실험을 소유한다.

## 구성

| 경로 | 역할 |
| --- | --- |
| `media/` | 그림·동영상 |
| `docs/` | 수식·실험 설계·측정·task 계약 |
| `tests/` | 환경 연결·rollout gradient·library 조립 검증 |
| `source/<task>/` | 관측·action·보상·종료·reference와 simulator 연결 |
| `source/robot/<robot>/` | 로봇 자산과 license |
| `scripts/` | 학습·평가·재생 실행 진입점 |

알고리즘은 `world_model_skills`의 공개 API에서 import한다. 수식·입력 약속·학습 계산은
[library 계약](https://github.com/CheolMin-Yoon/skills/blob/main/docs/world_model_skills/README.md)이 소유한다.
`source/ant/`는 결정론적 MLP·rollout·checkpoint 복원을 제공하고, `scripts/train.py`가 stock Rewarped Ant와
library `BPTT`를 조립한다.

## Ant 학습·재생·landscape

저장소 루트에서 활성화한 Rewarped 환경의 Python으로 실행한다.
관측·보상·종료·행동 clipping은 stock Ant를 사용하며, MLP는 Gaussian 샘플링이나 tanh 없이 행동을 출력한다.
학습은 고정 horizon의 BPTT와 Adam을 사용하고, 기본 gradient norm 상한은 1이다.
`source.ant.rollout`이 미분 가능한 보상과 실수 reset mask를 만들고, library가 episode reset마다 할인을 되돌려
손실을 `num_envs × horizon`으로 정규화한 뒤 정책을 갱신한다.

```bash
python -m scripts.train --output-dir logs/ant/bptt/example

python -m scripts.play \
  --checkpoint logs/ant/bptt/example/checkpoint_001000.pt \
  --renderer usd --output-dir media/play/example

python -m scripts.plot_loss_landscape \
  --checkpoints logs/ant/bptt/example/checkpoint_000000.pt logs/ant/bptt/example/checkpoint_001000.pt \
  --labels initial bptt --name ant_bptt
```

`train`의 `--num-envs`, `--horizon`, `--iterations`, `--lr`, `--hidden-dims`로 실행 크기와 정책을 정한다.
출력 폴더는 새 경로여야 하며, `config.json`, `metrics.csv`, 초기·주기별·마지막 checkpoint를 저장한다.
CSV의 loss는 갱신 전 할인 rollout 목적함수이고, episode return은 그 iteration에서 완료된 episode의
할인 없는 보상 합계 평균이다. 아직 종료된 episode가 없으면 해당 평균은 NaN이다.
Checkpoint는 정책 설정·state_dict·optimizer·iteration·환경 및 학습 설정을 포함한다.
Simulator·RNG 상태는 저장하지 않으며 학습 재개 옵션은 제공하지 않는다.

`play`는 고정 초기 상태에서 첫 episode를 재생한다. `--renderer none`은 수치 평가,
`usd`는 `AntDFlex_1.usd` 장면 기록, `opengl`은 화면 재생이다.
`--steps`로 제한한 구간이 종료에 도달하지 않으면 미종료 구간으로 표시한다.

Landscape는 같은 모델 구조의 checkpoint를 비교한다. 고정 초기 상태·공통 episode 길이에서 첫 종료까지의
할인 없는 return을 평가하고, 높이는 `symlog(-return)`으로 표시한다. 각 패널의 색 범위는 독립적이므로
색만으로 절대 성능을 비교하지 않는다. `--episode-length`로 공통 평가 길이를 지정할 수 있다.
`--grid-size`, `--batch-size`, `--radius`로 격자·동시 환경 수·parameter 변화 범위를 정한다.
그림은 `media/loss_landscape/`, 원값과 평가 조건은 `logs/loss_landscape/`에 저장한다.

## 환경과 검증

Python과 package 버전은 [pyproject.toml](pyproject.toml)과 `uv.lock`으로 고정한다.
실행 환경은 `$HOME/.venvs/rewarped`이며 GPU가 필요하다.
형제 경로 `../skills` checkout을 editable dependency로 사용한다.

```bash
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/rewarped" uv sync --locked --inexact
source "$HOME/.venvs/rewarped/bin/activate"
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

환경 검증은 렌더링 없이 Rewarped Ant 2개를 2 step 전진시키고 보상에서 action으로의 역전파를 확인한다.
학습 실행은 위 명령으로 사용자가 시작한다.
Rewarped의 버전 경계와 진단 메시지는 [환경 계약](docs/environment.md)을 따른다.
