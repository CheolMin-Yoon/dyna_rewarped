# Rewarped 실환경 검증

## Ant SHAC 100회: 원본과 world_model_skills 비교

원본 Mineral Ant SHAC 실행 중 사용자가 100회 비교를 요청했다. 요청 확인 시 이미 176회가 표시되어 SIGINT를
전달했고, 최종 완료 185회·378,880 전이에서 사용자 요청으로 중단됐다. 학습 실패가 아니며 재시작하지 않았다.
비교에는 보존된 `epochs100_steps204k_rewards1273.62.pth`만 사용했다(내부 epoch 99·204,800 전이).
원본의 최종 실행 checkpoint·평가는 사용자 중단으로 없지만 100회 checkpoint는 유한하다.

`world_model_skills.SHAC`·`PolicyModel`·`CriticModel`과 `torch_skills.RunningNormalization`을 쓰는 consumer를
새로 100회 실행했다. Stock Rewarped Ant를 직접 생성하며 물리 task·알고리즘 source는 이번 비교에서 바꾸지 않았다.

| 항목 | 맞춘 설정 |
| --- | --- |
| 정책·critic | `(128,64,32)` · `(64,64)` |
| Hidden·초기화 | Linear → LayerNorm → ELU, 직교 gain sqrt(2), bias 0 |
| 초기 표준편차 | exp(-1), 학습 가능 |
| 관측 정규화 | epsilon 1e-5, 초기 count 1e-4, rollout 직전 통계 동결 |
| Adam | 정책·critic LR 0.002, betas (0.7,0.95), epsilon 1e-8, weight decay 0 |
| LR 스케줄 | 원본의 2,000회 기준 선형 감소 유지, 종착값 1e-5 |
| 환경·rollout | 64환경, horizon 32, episode 1,000, randomize=True, no_env_offset=True |
| Seed | 기본 1000, setup 107955, train 856663 |
| SHAC | gamma 0.99, lambda 0.95, critic 16회×4 minibatch, target alpha 0.2, grad norm 1 |

Consumer는 110.58초에 100회·204,800 전이를 완료했다(exit 0). 초기와 100회 checkpoint의 모든 tensor가
유한했고, 모든 갱신의 정책·critic loss와 LR이 유한했다. 최종 RMS count 211,200=100×33×64,
정책·critic Adam step 100·6,400, 두 LR 0.001901495를 확인했다. 완료 episode가 없는 31개 CSV 행의
episode return·length NaN은 빈 통계이며 손실·물리 발산이 아니다.

학습 중 기록의 서로 다른 집계 방식을 직접 비교하지 않고 별도 평가했다. 각 정책을 독립 프로세스에서 복원하고,
평가 seed 71751·64환경·randomize=True·episode 제한 1,000·결정론적 tanh(mean)·공통 FP32/TF32를 적용했다.
Checkpoint별 관측 정규화 통계를 고정했다. 첫 episode만 환경마다 한 번 집계했으며, 세 조건의 초기 raw 관측과
관절 상태가 bitwise 동일함을 확인했다.

| 정책 | 평균 return ± 환경 간 표준편차 | 평균 길이 | 실패 종료 / 시간 제한 |
| --- | ---: | ---: | ---: |
| skills 초기 | 182.236 ± 157.002 | 415.859 | 61 / 3 |
| skills 100회 | 1,247.417 ± 146.309 | 985.891 | 1 / 63 |
| 원본 100회 | 1,266.449 ± 33.398 | 1,000 | 0 / 64 |

모든 평가 episode가 유한한 return으로 종료했다. skills의 학습 전후 개선을 확인했으며 원본 100회 대비
평균 return 차이는 −19.032(−1.50%)였다. 단일 학습 seed와 하나의 공통 초기 상태 batch 비교이므로
일반적인 성능 우열·동등성을 확정하지 않는다. 하이퍼파라미터는 맞췄지만 초기 가중치·난수 소비 순서·compile 경로,
원본의 pre-step bootstrap·shuffle 없는 critic dataset 등 구현 차이는 남아 있다.
Consumer의 모델 forward는 기본 compile·FP32/TF32이고 원본 학습은 수정하지 않은 runner 설정이다.
서로 다른 경로의 실행 시간을 compile 속도 비교로 해석하지 않는다.

원본 기록: `logs/stock_mineral_ant/20261107_010154_128326/`.
Consumer 기록: `logs/skills_ant_comparison/20261107_010612_125131/`의 `result.json`, `train/`,
`evaluation/summary.json`과 평가 명령·하네스·초기 상태 tensor. 두 실행은 모두 종료됐고 재개하지 않았다.
`hyperparameter_comparison.json`에는 config·checkpoint·source로 대조한 일치 39항목과 구현·실행 차이를 기록했다.

## 수정하지 않은 Mineral 원본 파이프라인 대조

Mineral `12a660435e6d71bf05872e5236234d90af02e6fd`의 원본 `python -m mineral.scripts.run`으로
Rewarped AllegroHand SHAC `train_eval`을 실행했다. [실행 명령](../README.md#원본-mineral-파이프라인-대조)은
upstream HandReorient recipe와 같고, W&B만 `online`에서 `disabled`로 바꿨다.
원본 checkout은 실행 전후 clean이며, 실행 경로는 Mineral의 SHAC·모델·분포다.
`world_model_skills`·`torch_skills`는 원본 source에 import되지 않고 이 consumer의 trainer도 사용하지 않았다.

사용자 승인 후 `stock` extra의 직접 의존성 7개·하위 의존성을 포함해 42개 패키지를 추가했다.
Dry-run과 설치에서 기존 패키지 변경·삭제는 0개였다. 실행 시 Torch 2.15.0·Warp 1.7.2·NumPy 1.26.4·Rewarped
8500e64를 유지했다. 원본 학습 코드는 수정하지 않았다.

| 항목 | 결과 |
| --- | --- |
| 실행 시간 | 5.03초 |
| 중단 지점 | 첫 actor update의 rollout, `Normal` 분포 생성 |
| 오류 | `loc [64,16]`에 NaN이 있어 `ValueError` |
| 완료 epoch | 0 |
| Checkpoint·평가 | checkpoint 0개, `final.pth`·`scores.json` 없음 |
| TensorBoard | config text만 존재, 학습 scalar 없음 |
| 프로세스 종료 코드 | 0, 원본 runner가 예외를 catch했으므로 학습 성공을 뜻하지 않음 |

따라서 현재 설치 환경에서는 library 교체 없이도 원본 학습 파이프라인이 실패한다.
이 결과로 실패를 `_skills` 구현에만 귀속할 수 없지만, library 전체가 원본과 동등하거나 무결하다는 의미도 아니다.
원본 로그에는 최초 NaN 전이 시각이 없어 simulator의 특정 연산·버전·GPU를 원인으로 확정하지 않는다.
실패한 학습은 재시작하지 않았다.

원본 기록은 `logs/stock_mineral_allegro/20261107_005503_258146/`의 `manifest.json`, `result.json`,
`stdout.log`, `run/resolved_config.yaml`과 TensorBoard event다.

### 정책 없는 첫 전이 대조

학습 재시작 없이 stock `AllegroHand`에 zero action을 한 번 넣었다. 각 subprocess에서
Mineral·`world_model_skills`·`torch_skills`는 import하지 않았고 모델·optimizer·역전파도 사용하지 않았다.
학습과 같은 `no_grad=False`, `use_graph_capture=True`, `no_env_offset=True`를 유지했다.

| 초기화 | 환경 수 | 초기 q·qd·obs·action | 첫 step 후 NaN 원소 수 q / qd / obs |
| --- | ---: | --- | --- |
| 원본 무작위화 | 64 | 모두 유한 | 432 / 432 / 864 |
| 원본 무작위화 | 1 | 모두 유한 | 0 / 0 / 0 |
| 고정 초기화 | 64 | 모두 유한 | 512 / 512 / 1,024 |
| 고정 초기화 | 1 | 모두 유한 | 0 / 0 / 0 |

원본 무작위화에서는 환경 수에 따른 난수 배열 크기 차이가 있으므로 원인을 환경 수 하나로 좁힐 수 없다.
이어 `randomize=False` 두 조건을 대조했고, 64개 환경 각각의 초기 `joint_q`·`joint_qd`·관측이
1환경의 값과 bitwise 동일함을 CPU 배열로 확인했다(max difference 0). 같은 zero action의 첫 step 후
64환경에서만 NaN이 생겼다. 이때 보상은 아직 유한하고 종료·reset 신호도 모두 False이므로 reset으로 가려진
결과가 아니다.

이 대조는 정책·정규화·학습 갱신·초기 난수 차이를 제거한 상태에서도 현재 설치의 다중환경 simulator 전진 경로가
비유한 상태를 만든다는 증거다. 특정 kernel·Warp/driver 호환성·충돌 처리 중 어느 것이 근본 원인인지는
아직 확정하지 않았다. 기본 `separate_collision_group_per_env=True`가 builder에 전달되므로
`no_env_offset=True`만으로 환경 간 충돌을 원인으로 단정하지 않는다.

원본 실행 폴더의 `diagnostics/`에 각 명령·하네스·로그·JSON·초기 상태 tensor와
`fixed_initial_comparison.json`을 저장했다. Production source·물리 설정·패키지 버전은 바꾸지 않았다.

## AllegroHand SHAC 원본 설정 검증

Rewarped용 HandReorient SHAC 예제의 설정으로 실행했지만 첫 rollout부터 보상이 비유한 값이 되어
첫 optimizer 갱신 전에 중단됐다. 실행 명령과 설정은 [README](../README.md#rewarped-allegrohand-shac)에 있다.
Stock 환경을 직접 사용하고 `world_model_skills.SHAC`·`PolicyModel`·`CriticModel`과
`torch_skills.RunningNormalization`을 consumer에서 연결했다. Library source는 이번 작업에서 변경하지 않았다.

- 64환경, horizon 32, episode 600, 목표 2,000회(4,096,000 전이), randomize=True
- 정책 `(512,256)`, critic `(256,256)`, LayerNorm → ELU, 직교 gain sqrt(2), 초기 std exp(-1)
- Adam betas `(0.7,0.95)`, 정책 LR 0.002·critic LR 0.0005, 2,000회 기준 선형 감소
- gamma 0.99·lambda 0.95, critic 16회×4 minibatch, target alpha 0.995, grad norm 1
- 기본 seed 1100, 원본 runner에서 파생한 setup seed 343430·train seed 374829 적용
- FP32/TF32, 모델 forward 기본 compile, 추가 손실 compile 비활성
- 관측 dict의 `obs` 72차원만 정책·critic에 전달, 행동 16차원, reset 전 관측의 미분 유지

사용자 승인 후 `hydra-core==1.3.2`, `omegaconf==2.3.1`, `antlr4-python3-runtime==4.9.3`,
`pyyaml==6.0.3`을 추가했다. `uv sync --dry-run --inexact --locked`로 기존 패키지 변경·삭제 0개를 확인한 뒤
설치했고 `pyproject.toml`·`uv.lock`에 반영했다. Torch·Warp·Rewarped 등 기존 버전은 유지했다.
연결·gradient·정규화·checkpoint·실패 자료 저장 CPU 검사 16개, Ruff·format·lock·diff 검사를 통과했다.

| 항목 | 결과 |
| --- | --- |
| 학습 종료 | 15.84초, exit 1, 첫 정책 loss 유한성 검사에서 중단 |
| 완료 갱신 | 0, 초기 checkpoint만 저장 |
| 첫 비유한 reward | 첫 전이 `t=0`, 환경 32; termination=0·reset=0 |
| Reward 유한성 | 2,048개 중 377개 유한, 1,671개 비유한 |
| Next observation | 첫 전이부터 49개 환경 비유한 |
| Policy 입력 관측 | 첫 비유한 값은 두 번째 전이 `t=1` |
| 초기 checkpoint | 모든 tensor 유한, optimizer state 비어 있음, RMS count 1e-4 |
| 초기 결정론 재생 | 같은 64환경에서 비유한 보상으로 exit 1, 완료 episode 결과 없음 |

실패한 rollout을 `failed_rollout.pt`에 저장했다. CPU에서 초기 target critic에 유한한 next observation 267개를
넣었을 때 출력은 전부 유한했다(-3.21961~3.26510). Bootstrap 곱셈에서 `0×NaN`도 발생했지만,
활성 bootstrap에 비유한 출력이 54개 있고 reward 자체가 이미 비유한 값이므로 종료 bootstrap을 0으로
처리하는 것만으로 이번 실패를 해결할 수 없다.

이번 실행의 직접 중단 조건은 학습된 가중치 발산이 아니라 첫 rollout의 비유한 입력이다. 초기 정책 파라미터가
유한하다는 사실만으로 simulator·행동 변환·환경 설정 중 근본 원인을 특정하지 않는다. 전이별 raw 물리 상태와
행동은 저장하지 않았고, 같은 설치 환경에서 원본 Mineral SHAC의 성공을 확인한 것도 아니다.
NaN 치환·환경 수 축소·물리 설정 변경·학습 재시작은 수행하지 않았다.

원본 기록은 `logs/stock_allegro_validation/20261107_004649_794207/`의 `manifest.json`, `result.json`,
`mask_diagnostic.json`, `train.log`, `train/failed_rollout.pt`, `train/checkpoint_000000.pt`에 있다.

## Humanoid 실환경 검증

Stock 하이퍼파라미터를 적용한 최신 SHAC 검증도 2회 갱신 후 3번째 정책 손실의 비유한 값으로 중단됐다.
이전 2026-11-06 기본 설정에서는 BPTT와 SHAC 모두 첫 갱신 후 두 번째 갱신에서 중단됐다.
작은 실행 검사의 통과만으로 이 조건의 연속 학습이 안정적이라고 판단할 수 없다.

## Stock Humanoid SHAC 설정 적용

Mineral revision `12a660435e6d71bf05872e5236234d90af02e6fd`의
[DFlex Humanoid SHAC 예제](https://github.com/etaoxing/mineral/blob/12a660435e6d71bf05872e5236234d90af02e6fd/docs/dflex.md#L274-L290)와
`mineral/cfgs/agent/DFlexAntSHAC.yaml` 기본값을 대조했다. 문서의 `actor_critic_kwargs`는 현재 source의
`actor_kwargs`·`critic_kwargs`와 이름이 달라, 문서가 지정한 모델 크기를 현재 consumer 인자로 전달했다.
원본 예제의 backend는 DFlex이고 이번 실행은 Rewarped Humanoid이므로 하이퍼파라미터 적용 검증이다.

| 항목 | 적용값 |
| --- | --- |
| 정책·critic hidden | `(256, 128)` · `(128, 128)` |
| 각 hidden block | Linear → LayerNorm → ELU |
| 초기화 | 모든 Linear 직교 gain `sqrt(2)`, bias 0 |
| 정책 표준편차 | 초기 `exp(-1)`, 학습 가능 |
| 관측 통계 | epsilon `1e-5`, 초기 count `1e-4`, rollout 직전 동결 |
| Adam | betas `(0.7, 0.95)`, epsilon `1e-8`, weight decay 0 |
| 정책·critic 초기 학습률 | `0.002` · `0.0005` |
| 학습률 스케줄 | 5,000회 기준으로 `1e-5`까지 선형 감소 |
| 환경·rollout | 64환경, horizon 32, episode 1,000, seed 110, randomize |
| SHAC | gamma `0.99`, lambda `0.95`, critic 16회 × 4 minibatch, grad norm 1 |
| Target critic | 기존 target 가중치 `0.995` |

Consumer에서 `RunningNormalization` 통계 누적·동결·checkpoint 복원과 LR 스케줄을 연결했다.
정책·critic·optimizer는 현재 editable `skills`의 공개 API를 사용한다. 실행 source는 미커밋 변경을 포함하며
각 파일 hash와 저장소 revision을 manifest에 기록했다. 패키지 버전은 변경하지 않았다.
추가 손실 compile은 끄고 모델 forward의 기본 compile·FP32/TF32를 유지했다.

200회 제한 검증으로 시작해 22.6초 후 exit 1로 종료했다. 원본 약 1,000만 전이 전체 학습은 수행하지 않았다.

| 항목 | 결과 |
| --- | --- |
| 완료 갱신 | 2 / 200, 환경 전이 4,096개 |
| 중단 지점 | 3번째 `SHAC.update`, 정책 loss 유한성 검사, backward 이전 |
| 마지막 정책·critic loss | −1.349350 · 328.231659 |
| 저장 checkpoint | iteration 0·1·2 |
| 마지막 checkpoint tensor | 정책·critic·target·optimizer·정규화 모두 유한 |
| 정규화 count | 4,224 = 2 × (32 + 1) × 64 |
| 정책·critic optimizer step | 2 · 128 |
| 두 번째 갱신 학습률 | `0.001999602` · `0.000499902` |
| 초기 정책 재생 | return 37.173, 15 step |
| 2회 학습 후 재생 | return 18.222, 16 step |

재생은 각 checkpoint의 정규화 통계를 복원하고 seed 110·64환경·`randomize=False`·결정론적 행동·episode 제한
1,000을 공통 적용했다. 같은 초기 상태를 복제한 평가이므로 64개 독립 seed 결과가 아니다. 재생은 모두 exit 0이나
성능 개선은 확인하지 못했다. 중단한 학습은 재시작하지 않았다.

하이퍼파라미터 일치와 stock 학습 경로 일치는 구분한다. 원본 SHAC는 `share_encoder=True`에서 step 전 latent를
next value에 쓰고, critic 표본을 shuffle하지 않으며, 비유한 critic gradient를 0으로 치환한다. Library는 다음 관측을
사용하고 minibatch를 섞으며 비유한 loss·gradient에서 중단한다. 원본은 종료 관측이 비정상이면 bootstrap을 명시적으로
0으로 덮지만, 현재 library의 곱셈 mask는 비유한 critic 출력을 제거하지 못한다(`0 * NaN = NaN`).
이 차이들은 자동으로 복제하거나 우회하지 않았다. 실패한 세 번째 rollout의 보상·관측·물리 상태를 보존하지 않았으므로
어느 항에서 먼저 비유한 값이 생겼는지, 이 차이가 이번 중단 원인인지는 확정할 수 없다.

연결 검증 15개(CPU 14개·기존 Ant CUDA smoke 1개), Ruff와 `git diff --check`를 통과했다.
명령·설정·source hash·종료 코드·원본 로그·checkpoint·재생 결과는
`logs/stock_humanoid_validation/20261107_003205_014357/`의 `manifest.json`, `result.json`, `train.log`, `train/`에 있다.

## 조건과 학습 결과

- Library: `skills` revision `496f30828efcbca11837d29d852b868e2e8722e9`
- 환경: `$HOME/.venvs/rewarped`, stock Rewarped Humanoid, GPU RTX 5080
- 64환경, horizon 32, episode 제한 1,000, seed 0, `randomize=True`, `no_env_offset=True`
- `PolicyModel` 64×64, 초기 표준편차 1.0, SHAC `CriticModel` 64×64
- FP32·TF32, 모델 `forward` compile 기본 mode, 추가 손실 compile 비활성
- 각각 200회 갱신 목표, checkpoint 저장 간격 25

| 항목 | BPTT | SHAC |
| --- | ---: | ---: |
| 완료 갱신 수 | 1 | 1 |
| 완료 갱신에 사용한 환경 전이 수 | 2,048 | 2,048 |
| 첫 정책 손실 | −1.47809 | −1.47978 |
| 첫 critic 손실 | 해당 없음 | 300.19 |
| 첫 구간 완료 episode 수 | 84 | 84 |
| 완료 episode 평균 return | 23.729 | 23.737 |
| 완료 episode 평균 길이 | 15.655 | 15.655 |
| 프로세스 종료 코드 | 1 | 1 |
| 최신 저장 checkpoint iteration | 0 | 0 |

두 learner의 비유한 손실 검사가 두 번째 backward·optimizer 갱신 전에 실행을 중단했다.
실패한 run은 재시작하지 않았다. 첫 갱신은 완료했지만 저장 간격 이전에 실패하여 학습 후 정책은 남아 있지 않다.
따라서 초기·학습 후 정책의 성능 향상, 수렴, 학습 후 landscape는 평가할 수 없다.
학습 CSV의 episode 평균은 해당 구간에서 종료된 episode만 포함하므로 환경별 동일 시도 수 평가와 구분한다.

## 물리 전진 계산 진단

초기 BPTT 정책을 고정하고 optimizer와 backward 없이 같은 크기의 rollout을 별도로 검사했다.
44번째 전이에서 환경 7의 `joint_q`, `joint_qd`, 관측과 보상에 처음 NaN이 발생했다.
직전 정책 입력은 유한했지만 관측 최댓값은 약 34,633까지 커져 있었고, 행동은 모두 유한한 `[-1, 1]` 범위였다.
Reset 전 관측 수집 시점에 이미 물리 상태가 NaN이었다.

이는 정책 갱신 없이도 현재 설정의 simulator forward가 비유한 값을 만들 수 있다는 재현이다.
원래 학습 실패와 정확히 동일한 전이·원인이라고 단정하거나, stock 환경 전체의 결함으로 일반화하지 않는다.
관측·초기 정책 분포·행동 설정·물리 적분의 어느 조건이 발산을 유발하는지는 추가 분리가 필요하다.
비유한 값을 가리는 처리나 simulator·library 소스 수정은 적용하지 않았다.

## 초기 정책 재생

`scripts.play`의 `deterministic=True`, `randomize=False` 조건에서 첫 episode를 재생했다.
64환경 수치 재생은 모두 22-step 종료, 평균 return 31.321이었다.
1환경 USD 재생도 22-step 종료, return 31.320으로 실행됐으며 장면 파일을 저장했다.
이는 초기 정책의 재생 경로 검증이며 stochastic 학습 안정성이나 학습 성과를 뜻하지 않는다.

## Compile 판정

양쪽 모델 `forward` compile은 유지하고, BPTT `loss` 또는 SHAC `loss`·`critic_targets`·`critic_loss` 묶음만
추가 compile한 경우를 비교했다. 초기 checkpoint의 모델·optimizer·RNG와 공통 초기 물리 상태를 매번 복원했다.
각 알고리즘에서 양쪽 warmup 3회씩, 같은 기본 경로 AA 대조 2회, AB/BA/AB 짝 비교 3쌍을 수행했다.
측정 구간은 실제 rollout과 전체 learner update이며 CUDA 동기화를 포함한다.

| 알고리즘 | 기본 경로 중앙값 | 추가 compile 중앙값 | 짝별 속도비 중앙값 | 판정 |
| --- | ---: | ---: | ---: | --- |
| BPTT | 4,210.52 ms | 4,209.27 ms | 0.9994배 | 이득 없음, 미채택 |
| SHAC | 4,245.15 ms | 4,238.94 ms | 1.0015배 | 약 0.15% 차이, 채택 근거 부족 |

각 중앙값의 비율과 짝별 비율의 중앙값은 다른 통계량이다.
각각 14회 실행에서 유한 값·초기 물리 상태 정확 복원 검사를 통과했고, warmup 이후 새 compile graph는 없었다.
같은 기본 경로 AA 반복에서도 보상 벡터의 상대 L2 차이가 BPTT 약 0.62%, SHAC 약 0.58%였다.
추가 compile 비교의 수치 차이를 전부 compile 탓으로 돌릴 수 없으며, 엄격한 수치 동등성을 확인한 결과도 아니다.
특히 SHAC의 작은 시간 차이만으로 decorator 채택을 정당화하지 않는다.

최초 시도는 첫 초기화와 후속 reset 사이 `joint_q` 최대 약 `6.15e-8` 차이로 측정 도구의 복원 검사에서 중단됐다.
초기화 완료 후 reset 상태를 공통 기준으로 캡처하도록 임시 도구를 수정했고, 원래 실패 기록도 보존했다.
이는 학습 재시작이나 production 소스 수정이 아니다.

연속 학습은 여전히 실패한 상태다. 이 결과는 초기 정책에서 환경을 복원한 단일 갱신 비교에 한정되며
학습 중 달라지는 상태 분포에 대한 성능·안정성 근거로 확대하지 않는다. Library decorator는 추가하지 않았다.

## 원본 기록

로컬 산출물은 `logs/training_validation/20261106_222328_605089/`에 있다.

- `summary.json`: 명령, 종료 코드, 마지막 완료 iteration, checkpoint, 학습 오류
- `bptt/`, `shac/`: 설정, CSV, 초기 checkpoint
- `diagnostics/forward_trace.json`: 최초 비유한 전이와 단계별 관측·행동·물리 상태 범위
- `diagnostics/play_none.log`, `diagnostics/play_usd.log`: 초기 정책 재생 결과
- `diagnostics/usd/HumanoidDFlex_1.usd`: 초기 정책 장면
- `benchmark_bptt_corrected.json`, `benchmark_shac_corrected.json`: 교정 후 반복 측정·수치 차이·source hash
- `benchmark_bptt.json`, `benchmark_shac.json`: 교정 전 측정 도구 복원 실패 기록

로그와 checkpoint는 Git 제외 산출물이므로 이 문서만으로 원본 파일이 배포되지는 않는다.
