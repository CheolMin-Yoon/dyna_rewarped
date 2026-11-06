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

## 미분 경계

Rewarped의 simulator 미분과 학습한 dynamics 모델의 미분은 서로 다른 전이 구현이다.
공유 library로 이관할 때는 목적함수·학습 계산과 task의 전이·reset·보상 연결을 분리한다.
MJLab용 조립은 해당 consumer가 소유한다.
