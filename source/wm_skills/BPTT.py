import jax
import jax.numpy as jnp


def bptt_loss(params, init_state, eps, policy, step, gamma=0.99):
    r"""
    BPTT 목적함수

    $$
    L(\theta) = -\frac{1}{N}
    \sum_{i=1}^{N}\sum_{t=0}^{H-1}\gamma^t r_{t,i}
    $$

    eps [H, N, A]: ε_0 … ε_{H−1}, 각 ε_t ~ N(0, I), eps[t]를 policy에 전달
    rewards [H, N]: 시점별·환경별 스칼라 보상
    discounts [H]: 시간 인덱스 t에 대한 할인
    returns [N]: 할인된 보상을 시간축으로 합산한 환경별 수익
    """

    def rollout(state0, t):
        action = policy(params, state0, eps[t])
        state1, reward = step(state0, action)
        return state1, reward

    final_state, rewards = jax.lax.scan(rollout, init_state, jnp.arange(eps.shape[0]))
    discounts = gamma ** jnp.arange(rewards.shape[0])
    returns = (discounts[:, None] * rewards).sum(axis=0)
    return -returns.mean(), final_state


def bptt_update(params, init_state, eps, policy, step, learning_rate, gamma=0.99):
    r"""
    정책 parameter의 SGD 1회 갱신

    $$
    g = \nabla_\theta L(\theta)
    $$
    $$
    \theta_{\mathrm{new}} = \theta - \alpha g
    $$
    value_and_grad: (미분할 함수)(해당 함수의 입력) --> 출력 ((loss, final_state), grads)
    tree.map:
             lambda: 파라미터에 적용할 계산
             params: parameter 인자 입력
             grads: gradient 인자 입력
    """
    (loss, final_state), grads = jax.value_and_grad(bptt_loss, has_aux=True)(
        params, init_state, eps, policy, step, gamma
    )
    params = jax.tree.map(
        lambda parameter, gradient: parameter - learning_rate * gradient,
        params,
        grads,
    )
    return params, final_state, loss


def policy(params, state0, eps):
    r"""
    관측 기반 tanh Gaussian 행동 생성
    $$
    \mu_t = o_t W + b,\quad \sigma = \exp(\ell)
    $$
    $$
    a_t = \tanh(\mu_t + \sigma \odot \epsilon_t)
    $$
    obs [N, O]: 환경별 관측
    params["weight"] [O, A]: 관측을 행동 평균으로 변환하는 가중치
    params["bias"] [A]: 행동 차원별 평균 편향
    params["log_std"] [A]: 상태와 무관하게 학습하는 행동 차원별 로그 표준편차
    eps [N, A]: 환경별·행동 차원별 표준정규 noise
    """
    obs = state0.obs
    mean = obs @ params["weight"] + params["bias"]
    std = jnp.exp(params["log_std"])
    return jnp.tanh(mean + std * eps)


def init_params(obs_dim, action_dim, init_std=0.1):
    r"""
    선형 tanh Gaussian 정책 parameter 초기화
    $$
    W = 0,\quad b = 0,\quad \ell = \log \sigma_0
    $$
    처음 μ = 0 → a ≈ tanh(σ_0 ε), 기본 자세 근처에서 작은 탐색으로 출발
    """
    return {
        "weight": jnp.zeros((obs_dim, action_dim)),
        "bias": jnp.zeros(action_dim),
        "log_std": jnp.full(action_dim, jnp.log(init_std)),
    }


def make_step(env):
    r"""
    env 하나의 step을 N개 batch로 묶어 rollout 약속 (state1, reward)의 전이 함수 step을 만들어 반환

    step: state0 [N], action [N, A] → state1 [N], reward [N]
    """
    batched = jax.vmap(env.step)

    def step(state0, action):
        state1 = batched(state0, action)
        return state1, state1.reward

    return step
