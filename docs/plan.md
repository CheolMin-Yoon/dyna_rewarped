# 계획: 방법론 베이스

어떤 task를 하든 아래 방법론을 베이스로 쓴다. 방법론은 `skills/world_model_skills`가 소유하고, task(로봇·물체·codec·reward·
simulator adapter)는 consumer가 둔다. 이 저장소는 Rewarped(미분 가능 simulator)에서 알고리즘을 검증하는 consumer다.

## 1. 논문

| 이름 | 저자·제목 | 발표처·연도 | 한 줄 | 미분 출처 | actor | 우리 역할 |
| --- | --- | --- | --- | --- | --- | --- |
| SHAC | Xu, Makoviychuk, Narang, Ramos, Matusik, Garg, Macklin, *Accelerated Policy Learning with Parallel Differentiable Simulation*, [arXiv 2204.07137](https://arxiv.org/abs/2204.07137) | **ICLR 2022** (학회, 2022.04) | 짧은 horizon BPTT + γ^h target critic bootstrap, TD(λ) critic | 미분 simulator | Gaussian, rsample | 바닥. `discount_ratios`·`SHAC` 구현 완료 |
| SAPO | Xing, Luk, Oh, *Stabilizing Reinforcement Learning in Differentiable Multiphysics Simulation*, [arXiv 2412.12089](https://arxiv.org/abs/2412.12089), Rewarped | **ICLR 2025** (학회, 2025.04) | SHAC + 최대 entropy(r + αℋ, α 자동 조절), critic 둘(actor 평균·목표 min), target critic 제거, 상태 의존 σ·LayerNorm·AdamW | 미분 simulator | tanh-Gaussian | Gaussian actor 대표, RFO의 비교군 |
| RFO | Zhong, Li, Wang, Huang(칭화 IIIS), *Reparameterization Flow Policy Optimization*, [arXiv 2602.03501](https://arxiv.org/abs/2602.03501) | **arXiv preprint 2026.02**, 발표처 미확인 | SHAC surrogate + flow 정책(K=4 Euler, dynamics와 같이 미분) + CFM 정규화 둘(past-data c_past, uniform c_uni), critic 둘 평균 | 미분 simulator | flow, pre-tanh CFM target | flow actor |
| DMO | Amigo, Khorrambakht, Chane-Sane, Mansard, Righetti, *First Order Model-Based RL through Decoupled Backpropagation*, [arXiv 2509.00215](https://arxiv.org/abs/2509.00215), [코드](https://github.com/machines-in-motion/DMO_code) | **CoRL 2025** (학회, PMLR 305, 2025.09) | 궤적은 simulator로, policy gradient는 학습한 미분 가능 모델로 backward(gradient swap). Go2 실기 | 학습 모델, forward는 실제 | Gaussian, rollout은 mean | 실제 simulator 경로의 1차 미분 |
| SuperMimic | Liao, Shao, Long, Zakka, Horowitz, Abbeel, Sreenath, *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*, [arXiv 2610.28480](https://arxiv.org/abs/2610.28480) | **CoRL 2026 oral** (학회, 2026) | 모션 라이브러리 전체의 공유 dynamics를 하나의 학습 world model로 fitting, 고정 모델 위 짧은 상상 rollout BPTT. 129,514 모션 중 실패 25 | 학습 모델, 상상 rollout | 결정론 | 상상 rollout 경로. 참고 코드: [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced), [supermimic-mjlab](https://github.com/YHFone/supermimic-mjlab) |

발표처는 전부 학회(ICLR·CoRL)이고 저널 게재본은 없다. RFO만 아직 preprint다.

## 2. 조합

세 축이 독립이다.

| 축 | 선택지 |
| --- | --- |
| 미분 출처 | 미분 simulator(Rewarped, 검증용) / 학습 모델 상상(SuperMimic) / 학습 모델 swap(DMO) |
| actor | Gaussian(SHAC·SAPO) / flow(RFO) |
| critic | 하나(SHAC) / 둘(SAPO mean·min, RFO 평균) |

실제 로봇 경로는 production simulator(MJLab)가 미분 불가라 학습 모델 축만 가능하다. 그래서 최종 조합은
(SuperMimic | DMO) × (SAPO | RFO)이고 바닥은 SHAC이다. 어느 쪽이 무거운 물체 접촉에 맞는지는 경험적으로 비교한다.

## 3. 구현 단위

| 방법 | `world_model_skills` | consumer | 검증 task |
| --- | --- | --- | --- |
| SHAC | `algorithms/shac.py`, `models/{policy,critic}_model.py`, `storage/replay_buffer.py` — 있음 | `rollout`, `scripts/train.py` — 이 저장소 | Rewarped Ant |
| RFO | `algorithms/rfo.py`(critic 둘 평균, CFM 항 둘, `ReplayBuffer.recent`로 D_recent), `models/flow_policy_model.py` — 모델은 있음 | rollout이 pre-tanh action도 쌓음 | Rewarped Ant, 논문 Table 수치 |
| SAPO | `algorithms/sapo.py`(entropy 항, α 갱신, critic 둘 mean/min), `PolicyModel`에 log π·상태 의존 σ | 같은 rollout | Rewarped Ant |
| SuperMimic | dynamics model, K-step fitting, `imagine`, runner — **모델 학습 반쪽**, DMO와 공유 | MJLab consumer: state codec, 해석적 reward·종료, `Task` protocol | G1 추적 재현 |
| DMO | `swap_gradient`, 실제 step rollout 변형, 1-step fitting | MJLab consumer | Go2 보행 재현 |

## 4. 순서

1. RFO — `rfo.py` + 이 저장소의 rollout·train 분기. 파일 하나로 끝나고 논문 수치와 바로 비교된다.
2. SAPO — 작다. RFO와 같은 Ant에서 flow vs Gaussian 비교군.
3. 모델 학습 반쪽 — replay는 있고 dynamics model·fitting·`imagine`·runner. SuperMimic G1 재현으로 검증.
4. DMO — 3 위에 함수 둘. Go2 재현.
5. 새 task — 바퀴 베이스 양팔 휴머노이드의 무거운 물체 조작. consumer 하나 추가.

## 5. 선결 결정

- Python 경계: Rewarped env 3.11(usd-core 24.5) vs skills `>=3.12`. 이 저장소가 library를 import하려면 skills를 3.11로
  내리거나 Rewarped를 3.12로 올려야 한다. 그 전까지 알고리즘 파일은 양쪽에 사본으로 두고 같은 손 예제 test로 묶는다.
- 로봇·물체 자산: 5단계 전에 MJCF, 바퀴·접촉 모델링, 물체 codec을 정한다.
- NaN: 손실 함수는 비유한 값을 다루지 않는다. rollout 경계에서 중단하고 env 수로 피한다([환경 계약](environment.md)).
