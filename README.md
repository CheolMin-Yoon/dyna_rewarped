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
`source/dflex/`는 미분 rollout·checkpoint 복원을 제공하고, `source/dflex/train.py`가 stock Rewarped Humanoid·Ant와
library `BPTT`·`SHAC`·`PolicyModel`·`CriticModel`을 조립한다. `scripts/train.py`는 CLI 진입점이다. 방법론 베이스(SHAC·SAPO·RFO·DMO·SuperMimic)와 구현 순서는 [계획](docs/plan.md)에 있다.

## Stock Humanoid·Ant 학습·재생·landscape

저장소 루트에서 활성화한 Rewarped 환경의 Python으로 실행한다.
환경은 설치된 Rewarped의 `Humanoid`·`Ant`·`AllegroHand` 클래스를 직접 사용하며 새 task는 정의하지 않는다.
관측·보상·종료·행동 clipping은 stock 환경을 사용하며, stock 설정처럼 `no_env_offset=True`로 생성한다.
두 learner는 `PolicyModel`의 tanh-Gaussian 행동으로 학습하고, SHAC는 단일 `CriticModel`과 target critic을 사용한다.
모델 LayerNorm·직교 초기화·관측 정규화는 CLI로 선택한다. 기본값은 정규화 없음이며,
Mineral의 DFlex Humanoid SHAC 하이퍼파라미터를 적용해도 simulator와 학습 구현까지 동일해지는 것은 아니다.
`source.dflex.rollout`은 미분 가능한 보상과 실수 reset mask를 수집하고, SHAC에서는 실패 종료와 reset 전 `next_obs`까지 수집한다.
Library가 손실·선택한 optimizer·gradient norm clipping·SHAC target 갱신을 소유한다.

```bash
python -m scripts.train --algorithm bptt --output-dir logs/humanoid/bptt/example
python -m scripts.train --algorithm shac --output-dir logs/humanoid/shac/example

python -m scripts.play \
  --checkpoint logs/humanoid/bptt/example/checkpoint_001000.pt \
  --renderer usd --output-dir media/play/example

python -m scripts.plot_loss_landscape \
  --checkpoints logs/humanoid/bptt/example/checkpoint_000000.pt logs/humanoid/bptt/example/checkpoint_001000.pt \
  --labels initial bptt --name humanoid_bptt
```

`train`은 기본 Humanoid·BPTT이며 `--task ant`, `--algorithm shac`로 바꾼다. 기존 raw MLP Ant checkpoint도 재생할 수 있다.
`train`의 `--num-envs`, `--horizon`, `--iterations`, `--lr`, `--hidden-dims`로 실행 크기와 정책을 정한다.
`--init-std`는 학습 가능한 표준편차의 초기값이다. SHAC는 `--critic-hidden-dims`, `--critic-lr`,
`--critic-iterations`, `--num-mini-batches`, `--lam`, `--target-critic-alpha`로 critic 갱신을 정한다.
`--activation`, `--norm-position`, `--orthogonal-gain`은 정책과 critic에 공통 적용한다.
`--optimizer`는 `adam`·`adamw`·`sgd`·`rmsprop`, `--optimizer-kwargs`는 해당 PyTorch optimizer의 JSON 인자다.
`--lr-schedule linear --schedule-iterations 5000 --min-lr 1e-5`는 첫 갱신의 초기 학습률을 유지한 뒤
정책·critic 각각의 초기값에서 선형 감소한다. 실행 횟수와 스케줄 길이는 독립적이다.

`--normalize-obs`는 `torch_skills.RunningNormalization`을 사용한다. `--normalization-eps`와
`--normalization-initial-count`로 초기 통계를 정한다. Rollout 직전 통계를 복제해 구간 전체의 정책·critic 입력에
고정하고, 별도의 누적 통계는 구간 시작 관측과 각 step의 반환 관측으로 갱신한다. Reset 전 bootstrap 관측도
같은 고정 통계를 사용한다. 정규화는 정책 입력에만 적용하며 simulator 상태는 바꾸지 않는다.
출력 폴더는 새 경로여야 하며, `config.json`, `metrics.csv`, 초기·주기별·마지막 checkpoint를 저장한다.
CSV의 loss는 갱신 전 할인 rollout 목적함수이고, episode return은 그 iteration에서 완료된 episode의
할인 없는 보상 합계 평균이다. 아직 종료된 episode가 없으면 해당 평균은 NaN이다.
`critic_loss` 열은 SHAC의 평균 critic 손실이며 BPTT에서는 비워 둔다. `policy_lr`·`critic_lr`는 해당 갱신의 학습률이다.
Checkpoint는 정책 종류·설정·state_dict·optimizer·iteration·환경 및 학습 설정을 포함한다.
SHAC는 critic 설정·가중치·target critic·두 optimizer 상태도 저장한다.
관측 정규화를 사용하면 설정과 누적 통계도 저장하며, 재생·landscape에서 각 checkpoint의 통계를 복원해 고정한다.
Simulator·RNG 상태는 저장하지 않으며 학습 재개 옵션은 제공하지 않는다.
Learner 갱신에서 비유한 값·실행 오류가 발생하면 `failed_rollout.pt`에 iteration과 입력 batch를 저장한 뒤 오류를 전파한다.

`play`는 고정 초기 상태에서 `deterministic=True`의 평균 행동으로 첫 episode를 재생한다. `--renderer none`은 수치 평가,
`usd`는 `<Task>DFlex_<환경 수>.usd` 장면 기록, `opengl`은 화면 재생이다.
`--steps`로 제한한 구간이 종료에 도달하지 않으면 미종료 구간으로 표시한다.

재생·landscape의 환경은 checkpoint의 task에서 선택한다. Landscape의 `--env`는 명시적 환경 재지정이다.
Landscape는 같은 task·모델 구조의 checkpoint를 결정론적 행동으로 비교한다. 첫 checkpoint의 환경 설정을 공통으로 사용한다. 고정 초기 상태·공통 episode 길이에서 첫 종료까지의
할인 없는 return을 평가하고, 높이는 `symlog(-return)`으로 표시한다. 각 패널의 색 범위는 독립적이므로
색만으로 절대 성능을 비교하지 않는다. `--episode-length`로 공통 평가 길이를 지정할 수 있다.
`--grid-size`, `--batch-size`, `--radius`로 격자·동시 환경 수·parameter 변화 범위를 정한다. 기본 batch는 64다.
그림은 `media/loss_landscape/`, 원값과 평가 조건은 `logs/loss_landscape/`에 저장한다.

## Rewarped AllegroHand SHAC

[Mineral HandReorient SHAC 예제](https://github.com/etaoxing/mineral/blob/12a660435e6d71bf05872e5236234d90af02e6fd/docs/rewarped.md#L167-L183)의
하이퍼파라미터를 아래처럼 전달한다. 설치된 `isaacgymenvs.allegro_hand.AllegroHand`를 직접 사용한다.
이 이름은 Rewarped 내부 suite 경로이며 별도의 Isaac Lab 설치를 뜻하지 않는다.
환경 생성에는 `hydra-core`와 `omegaconf`가 필요하다. 패키지 추가는 [환경 규칙](AGENTS.md)을 따른다.

```bash
python -m scripts.train --task allegro_hand --algorithm shac \
  --num-envs 64 --horizon 32 --episode-length 600 --iterations 2000 \
  --seed 1100 --setup-seed 343430 --train-seed 374829 --randomize \
  --hidden-dims 512 256 --critic-hidden-dims 256 256 \
  --activation elu --norm-position all --orthogonal-gain 1.4142135623730951 \
  --init-std 0.36787944117144233 --normalize-obs \
  --normalization-eps 1e-5 --normalization-initial-count 0.0001 \
  --optimizer adam --optimizer-kwargs '{"betas":[0.7,0.95],"eps":1e-8,"weight_decay":0.0}' \
  --lr 0.002 --critic-lr 0.0005 --lr-schedule linear --schedule-iterations 2000 --min-lr 1e-5 \
  --gamma 0.99 --lam 0.95 --max-grad-norm 1 --critic-iterations 16 --num-mini-batches 4 \
  --target-critic-alpha 0.995 --save-interval 1 --log-interval 50 --no-compile-losses \
  --output-dir logs/allegro_hand/shac/stock_recipe
```

`--seed`는 환경 seed, `--setup-seed`는 모델 구성 전 RNG, `--train-seed`는 첫 학습 reset 전 RNG다.
위 두 값은 원본 runner의 `RandomState(1100)` 파생값이다. 미지정하면 기존 seed 동작을 유지한다.
모델 생성 순서·compile 난수 경로까지 원본과 동일하다는 의미는 아니다.
기본 episode 길이는 AllegroHand 600, Ant·Humanoid 1,000이며 명시 인자로 덮어쓸 수 있다.
AllegroHand의 72차원 `obs`를 dict에서 꺼내 정책·critic·정규화에 전달하고, 16차원 행동을 stock 환경에 넘긴다.
물리 시간 간격·substep·보상·종료·torque 변환은 stock 환경이 소유한다.
원본과 library의 bootstrap·minibatch 순서·비유한 값 처리 차이는 [검증 기록](docs/validation.md)에 있다.

## 원본 Mineral 파이프라인 대조

`stock` extra는 Mineral의 로깅·수치 의존성이다. 사용자 승인 후 아래 명령으로 설치하며 기존 Rewarped 환경을 사용한다.

```bash
UV_PROJECT_ENVIRONMENT="$HOME/.venvs/rewarped" uv sync --locked --inexact --extra stock
```

Mineral revision `12a660435e6d71bf05872e5236234d90af02e6fd`의 수정하지 않은 checkout에서 실행한다.
아래 명령은 이 consumer·`world_model_skills`를 호출하지 않으며, 원본 recipe에서 W&B 기록만 비활성화한다.
`logdir`에는 새 절대 경로를 지정한다.

```bash
python -m mineral.scripts.run \
  task=Rewarped agent=DFlexAntSHAC task.env.env_name=AllegroHand task.env.env_suite=isaacgymenvs \
  logdir=/home/frlab/dyna_rewarped/logs/stock_mineral_allegro/example \
  agent.shac.max_epochs=2000 agent.shac.max_agent_steps=4.1e6 \
  'agent.network.actor_kwargs.mlp_kwargs.units=[512,256]' \
  'agent.network.critic_kwargs.mlp_kwargs.units=[256,256]' \
  agent.shac.critic_optim_kwargs.lr=5e-4 agent.shac.target_critic_alpha=0.995 \
  wandb.mode=disabled wandb.project=rewarped run=train_eval seed=1100
```

원본 runner는 예외를 출력하고 정상 종료 코드로 끝낼 수 있다. 성공 여부는 종료 코드와 함께 traceback,
마지막 epoch, `ckpt/final.pth`, 평가 `scores.json`을 확인한다. 실행 결과는 [검증 기록](docs/validation.md)에 둔다.

## Compile 비교 조건

학습은 FP32·TF32이며 모델 `forward`에 `torch.compile` 기본 mode를 적용한다.
`--compile-losses`를 지정하면 BPTT는 `loss`, SHAC는 `loss`·`critic_targets`·`critic_loss`를 추가로 compile한다.
이 옵션은 consumer에서 후보를 시험하는 용도이며 library의 decorator 채택을 의미하지 않는다.
`config.json`에 PyTorch 버전·정밀도·compile 선택·모델 및 환경 설정을 기록한다.

```bash
python -m scripts.train --algorithm shac --compile-losses --output-dir logs/humanoid/shac/compile_candidate
```

속도 비교에서는 양쪽의 모델 compile·정밀도·환경과 tensor shape를 같게 두고, 초기 compile을 제외한
rollout·역전파·optimizer 시간을 같은 상태·RNG로 복원한 짝 비교로 측정한다.
서로 다른 두 학습 run의 총 실행시간만으로 compile 이득을 판정하지 않는다.
짧은 실행 검사는 시작 가능 여부만 확인하며 장기 수렴·정책 품질·compile 성능 채택은 별도 검증한다.
Humanoid 64환경 연속 학습의 중단 원인과 재생·compile 판정은 [실환경 검증](docs/validation.md)에 기록한다.

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
