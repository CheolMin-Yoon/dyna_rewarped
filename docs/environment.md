# 환경 계약

## 실행 환경

실행 환경은 `$HOME/.venvs/rewarped` 하나다. Python·PyTorch·Rewarped·Warp 버전과 Rewarped source revision은
[pyproject.toml](../pyproject.toml)과 `uv.lock`이 소유한다.

Rewarped는 NVIDIA Warp의 물리 전진·역전파를 PyTorch autograd에 연결한다.
설치된 공식 checkout은 `$HOME/.venvs/rewarped/src/rewarped`에서 열람할 수 있다.
Package 설치·동기화에는 저장소의 lock을 사용한다.

- [Rewarped 공식 저장소와 설치 안내](https://github.com/rewarped/rewarped)
- [Rewarped·Warp 버전 대응](https://github.com/rewarped/rewarped#versioning)

## GPU 검증

`tests/test_rewarped.py`는 기본 CUDA graph capture를 켠 Ant 환경에서 2개 환경·2 step을 전개한다.
관측·보상·action gradient의 shape와 유한성을 확인하고 gradient가 0이 아닌지 검사한다.
렌더링이나 optimizer 갱신은 포함하지 않는다.

RTX 5080과 현재 드라이버에서는 Warp 초기화 중 `cuDeviceGetUuid` entry point 진단이 출력된다.
이 메시지가 물리·역전파 검사를 대신하지 않으며, 실행 가능 여부는 위 검사 결과로 확인한다.
Gym과 TorchScript의 upstream deprecation 메시지도 설치 검사에서 출력될 수 있다.

## 병렬 env 수

RTX 5080, Warp 1.7.2, Rewarped `8500e64`의 dflex Ant에서 env 수를 늘리면 일부 env가 첫 step부터 NaN 보상·관측을
낸다. 같은 정책, `randomize=False`, `no_grad=True`에서 25·128개는 0건(128개 200 step), 200개는 1건, 256개는 18건,
441개는 182건, 961개는 730건이었다. 깨진 env는 index에 흩어져 있고 CUDA graph capture를 꺼도 같다. 원인은 확인하지
않았으며, 그 전까지 병렬 평가와 학습은 env 128개 이하로 나눠 실행한다.

Humanoid는 같은 설치에서 128개 환경의 첫 step, 96개 환경의 16번째 step에 비유한 값이 발생했다.
64개 환경·horizon 32의 초기 rollout·역전파는 통과했지만 장기 학습 안정성을 보장하는 값은 아니다.
학습과 landscape의 기본 환경 수는 64다. 측정 원본은 `logs/benchmarks/compile_humanoid_20261106/`에 있다.

## skills와의 Python 경계

Rewarped `8500e64`가 고정한 `usd-core==24.5`에는 Python 3.12 wheel이 없어 이 환경은 Python 3.11이다.
`skills`는 Python 3.11 이상을 지원하며, 형제 checkout `../skills`를 editable dependency로 설치한다.
Rewarped 환경의 Python·PyTorch·Warp·USD 버전은 유지하고 MJLab·RSL-RL extra는 설치하지 않는다.
학습 계산은 설치된 `world_model_skills`에서 import하며, stock Humanoid·Ant의 정책 조립과 checkpoint 복원은 `source/dflex/`가 소유한다.

## 미분 경계

Rewarped의 simulator 미분과 학습한 dynamics 모델의 미분은 서로 다른 전이 구현이다.
목적함수·학습 계산은 `world_model_skills`에서 import하고, task의 전이·reset·보상 연결과 미분 가능한 rollout은
이 consumer가 소유한다. MJLab용 조립은 해당 consumer가 소유한다.
