# Neural Harmonic Textures 상세 분석 (한국어)

> 이 문서는 [nv-tlabs/neural-harmonic-textures](https://github.com/nv-tlabs/neural-harmonic-textures) 프로젝트의 알고리즘, 스크립트 실행 흐름, CUDA 커널을 상세하게 풀어 설명한 한국어 기술 문서입니다. 논문 *Neural Harmonic Textures for High-Quality Primitive Based Neural Reconstruction* (Condor et al., 2026) 의 내용을 구현 관점에서 읽어냅니다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [Neural Harmonic Textures의 물리적 의미](#2-neural-harmonic-textures의-물리적-의미)
3. [전체 아키텍처](#3-전체-아키텍처)
4. [핵심 알고리즘: 4단계 파이프라인](#4-핵심-알고리즘-4단계-파이프라인)
5. [사면체(Tetrahedron) 가상 Scaffold](#5-사면체tetrahedron-가상-scaffold)
6. [Harmonic Encoding (주기적 활성화)](#6-harmonic-encoding-주기적-활성화)
7. [Deferred Shading MLP 디코더](#7-deferred-shading-mlp-디코더)
8. [스크립트 실행 파이프라인](#8-스크립트-실행-파이프라인)
9. [CUDA 래스터라이저 NHT 커널 분석](#9-cuda-래스터라이저-nht-커널-분석)
10. [역전파 (Backward Pass)](#10-역전파-backward-pass)
11. [MCMC 밀도화 전략](#11-mcmc-밀도화-전략)
12. [AOV 확장 (멀티헤드 추론)](#12-aov-확장-멀티헤드-추론)
13. [하이퍼파라미터 설명](#13-하이퍼파라미터-설명)
14. [참고 자료](#14-참고-자료)

---

## 1. 프로젝트 개요

**Neural Harmonic Textures (NHT)** 는 NVIDIA Research와 USI Lugano가 공동 개발한 신경 표현 기법으로, **프리미티브 기반**(3D Gaussian 등) 표현과 **신경 필드**(NeRF, Instant-NGP 등) 표현의 장점을 동시에 취하기 위해 제안되었습니다.

| 구분 | 기존 3DGS | Neural Harmonic Textures (NHT) |
|------|-----------|--------------------------------|
| **색상 표현** | Spherical Harmonics (16개 × RGB) | 잠재 벡터 4개 × MLP 디코더 |
| **고주파 디테일** | 프리미티브 수로만 대응 → 한계 | 주기적 활성화로 세밀한 디테일 표현 |
| **셰이딩** | 파티클마다 per-ray 평가 | 래스터라이즈 후 **deferred MLP 1회** |
| **표현 성격** | Lagrangian(입자별 고정 SH) | Lagrangian + Fourier 기반 특징 혼합 |
| **M360 PSNR** | 27.94 | **28.63** (동일 primitive 수) |

핵심 아이디어는 다음 세 문장으로 정리됩니다.

1. 각 프리미티브(Gaussian) 주변에 **가상 사면체 scaffold** 를 두고, 그 4개 꼭짓점에 잠재 특징 벡터를 고정(anchor)합니다.
2. 광선과 Gaussian의 교점에서 4개 꼭짓점 특징을 **barycentric 보간**으로 구하고, 그 값에 `sin/cos` 같은 **주기적 활성화**를 적용합니다.
3. 여러 Gaussian의 alpha blending 결과는 자연스럽게 **조화 성분의 가중합**이 되며, 마지막에 **단 한 번의 deferred MLP** 로 RGB로 디코딩됩니다.

결과적으로 "정수 차수 SH 대신 연속 주파수 Fourier 기반 텍스처"를 각 프리미티브에 얹는 효과가 나고, 이는 **NeRF의 positional encoding에 대한 Lagrangian(입자) 관점 대안** 으로 해석됩니다.

---

## 2. Neural Harmonic Textures의 물리적 의미

### 2.1 왜 "Harmonic"이고 왜 "Texture"인가

기존 3DGS의 렌더링 방정식은 불투명도 α_i 와 색상 c_i 의 앞→뒤 알파 합성입니다.

```
C(x) = Σ_i  α_i T_i c_i,   T_i = Π_{j<i} (1 − α_j)
```

NHT는 `c_i` 자리에 **조화 성분** 을 집어넣습니다. 각 Gaussian에 대해 교차점에서 얻어진 특징 스칼라 b_i 에 대해

```
harmonic(b) = [ sin(1·b), cos(1·b), sin(2·b), cos(2·b), ... ]
```

가 렌더링되므로, alpha blending 이후 화면에서 모이는 값은

```
F_k(x) = Σ_i  α_i T_i  sin(k · b_i)     (cos 성분 대칭)
```

즉 **픽셀별로 축적된 주파수 k에 대한 조화 계수** 가 됩니다. 이 누적값을 MLP 로 디코딩하면 고주파 색상 신호를 만들 수 있습니다. 이것이 "Harmonic" 이라는 이름의 근거이자, NeRF의 positional encoding `[sin(2^k πx), cos(2^k πx)]` 와 수학적으로 같은 계열의 표현이라는 뜻입니다.

"Texture" 는 각 Gaussian 주위에 4 꼭짓점 scaffold가 있고, 그 꼭짓점에 올려진 잠재 벡터 4개를 보간한다는 점에서, 프리미티브 내부에 펼쳐진 **3D 텍스처 맵** 처럼 동작하기 때문에 붙은 이름입니다.

### 2.2 Lagrangian 해석

NeRF는 공간의 모든 점 x 에 대해 MLP(x) 로 필드를 정의하는 **Eulerian** 관점입니다. NHT는 특징을 "파티클이 움직이면 함께 움직이는" 꼭짓점 scaffold 에 부착하므로 **Lagrangian** 관점입니다. 즉,

- NeRF: "공간이 학습하고, 카메라는 그 공간을 샘플링한다"
- 3DGS + SH: "입자가 고정된 harmonic basis(SH)를 가진다"
- **NHT**: "입자가 학습 가능한 Fourier 계수를 들고 다닌다"

로 위계를 정리할 수 있습니다.

---

## 3. 전체 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                     학습 입력                             │
│    COLMAP/MipNeRF-360 형식 장면 (이미지 + 카메라 포즈)    │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│               Gaussian 프리미티브 상태                    │
│   means [N,3], quats [N,4], scales [N,3], opac [N]       │
│   features [N, F]  ← NHT 핵심: 프리미티브당 F차원 특징   │
└──────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴───────────────────┐
        ▼                                     ▼
┌───────────────────┐              ┌──────────────────────┐
│  CUDA 래스터라이저  │              │ MCMC Densification  │
│  (NHT 전용 커널)   │              │ (학습 중 primitive  │
│                   │              │  재배치/추가)        │
│ ① UT 프로젝션      │              └──────────────────────┘
│ ② 타일 할당        │
│ ③ Radix Sort       │
│ ④ NHT 픽셀 커널    │
│   ├ 사면체 보간    │
│   ├ harmonic enc.  │
│   └ alpha blend    │
└───────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│   렌더 텐서  (C, H, W, F·ENCF + 3)                       │
│     F·ENCF 채널 = 누적된 [sin, cos] 특징                 │
│     3 채널      = 각 픽셀의 광선 방향 (또는 타일 중심선)  │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────┐
│       Deferred Shader MLP (tcnn FullyFusedMLP)           │
│  입력: [조화 특징 | 광선 방향 SH 인코딩]                  │
│  출력: RGB (이미지당 1회, per-pixel loop 대신 batch)     │
└──────────────────────────────────────────────────────────┘
                          │
                          ▼
           Loss (L1 + D-SSIM + LPIPS) → 역전파
```

**핵심 파일 위치**:

| 구성요소 | 파일 |
|----------|------|
| 학습 진입점 | `gsplat/examples/simple_trainer_nht.py` |
| 뷰어 진입점 | `gsplat/examples/simple_viewer_nht.py` |
| NHT CUDA 전방 커널 | `gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorldNHT3DGSFwd.cu` |
| NHT CUDA 역전파 커널 | `gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorldNHT3DGSBwd.cu` |
| 사면체 보간/조화 인코딩 | `gsplat/gsplat/cuda/csrc/Interpolation.cuh` |
| Deferred Shader MLP | `gsplat/gsplat/nht/deferred_shader.py` |
| 밀도화 전략 | `gsplat/gsplat/nht/strategy.py` |
| 쉘 스크립트 | `scripts/{train,eval,view,download_data}.sh` |
| 벤치마크 스크립트 | `benchmarks/nht/benchmark_nht*.sh`, `benchmarks/benchmark_nht.py` |
| AOV(멀티헤드) 확장 | `aov/deferred_shader.py`, `aov/examples/simple_trainer_nht_aov.py` |

---

## 4. 핵심 알고리즘: 4단계 파이프라인

기존 3DGS의 4단계 렌더링 (프로젝션 → 타일 할당 → 정렬 → 픽셀 합성) 은 그대로 유지되며, **픽셀 합성 단계 내부의 "색상 조회" 부분만 NHT 로 교체** 됩니다. 따라서 NHT는 엄밀히 말해 **셰이딩 모델의 변경** 이지 토폴로지의 변경이 아닙니다.

```
[공통 단계]                        [NHT 전용 변경점]

Stage 1. 프로젝션 (UT)              파라미터 변화 없음
Stage 2. 타일 확장                  파라미터 변화 없음
Stage 3. Radix Sort                 파라미터 변화 없음
Stage 4. 픽셀 셰이딩                ── 여기서부터 NHT 전용 ──
          for each Gaussian hit:
            t* = −dot(g·r_o, g·r_d) / ‖g·r_d‖² (중심점 가까움)
            sample_pos = gro + t* · grd
            α = opacity · exp(−0.5 · ‖sample_pos‖²)

            # NHT 사면체 보간
            (w0,w1,w2,w3) = tet_barycentric_weights(sample_pos)
            b_k = Σ_v w_v · features[v, k]  ,  k = 0..F/4−1

            # 주기적(조화) 활성화
            for each freq f = 1..NUM_ENCODING_FREQUENCIES:
              h_sin[k,f] = sin(f · b_k)
              h_cos[k,f] = cos(f · b_k)

            pix_out += (α · T) · [h_sin, h_cos]
            T       *= (1 − α)

Stage 5. Deferred MLP (이미지당 1회)
          rgb = MLP( concat(pix_out, ray_dir_encoding) )
```

핵심 포인트:

- 프리미티브당 **F 개의 잠재 특징** 이 4 꼭짓점에 분할 저장됩니다. `F=48` 이면 꼭짓점 당 12차원.
- sin/cos 두 채널로 확장되므로 실제 픽셀 누적 텐서 폭은 `F/4 · ENCF`. 기본 설정(`NUM_ENCODING_FREQUENCIES=1`)에서 `ENCF = 2` 이므로 **F=48 → 24 채널** 이 픽셀 누적됩니다.
- 픽셀 마지막에 **ray 방향 3채널** 이 뒤따라 저장되어 deferred MLP 가 시점 의존성을 결정합니다.

---

## 5. 사면체(Tetrahedron) 가상 Scaffold

**파일**: `gsplat/gsplat/cuda/csrc/Interpolation.cuh`

각 Gaussian 에는 모양과 위치에 관계없이 **정규화된 inradius=1 짜리 정사면체** 4 꼭짓점이 붙어 있다고 가정합니다. 꼭짓점 좌표는 하드코딩되어 있습니다.

```
p0 = ( √6, −√2, −1)
p1 = (−√6, −√2, −1)
p2 = ( 0 ,  2√2, −1)
p3 = ( 0 ,  0 ,  3)
```

이 4 꼭짓점에 프리미티브의 F차원 특징이 **4등분** 되어 저장됩니다 (`F / VERTEX_PER_PRIM = F / 4` 차원/꼭짓점).

### 5.1 Barycentric 가중치 계산

광선-Gaussian 교차점을 **Gaussian 로컬 좌표계**(평균 = 0, 공분산 = I)로 변환한 점을 `sample_pos` 라 하면, barycentric 가중치는 4개 면에 대한 부호 있는 거리/높이로 주어집니다.

```cpp
// Interpolation.cuh (요약)
tet_barycentric_weights(sample_pos, w0, w1, w2, w3) {
    // n_i: 안쪽을 향한 i번째 면의 법선 (단위벡터, 상수)
    // h_i: 해당 면에서 맞은편 꼭짓점까지 높이 = 4
    w_i = dot(n_i, sample_pos − ref_i) / h_i
}
```

`w0 + w1 + w2 + w3 = 1` 이 수학적으로 보장되며, sample_pos 가 사면체 내부일 때 모두 [0, 1] 입니다. Gaussian 중심(0,0,0) 에서 각 `w_i = 1/4` 이 됩니다.

사면체는 한 번 계산된 상수(`TET_N0..3`, `TET_INV_H=0.25`) 로 구현되므로 런타임 비용이 매우 낮습니다. **4개 꼭짓점을 쓰는 이유** 는 3D 내부 점을 선형 보간으로 유일하게 복원할 수 있는 최소 조합이 tetrahedron(사면체)이기 때문입니다.

### 5.2 특징 보간

```cpp
// 꼭짓점 v 에 저장된 특징 길이 = CDIM / 4
result[k] = w0·v0[k] + w1·v1[k] + w2·v2[k] + w3·v3[k]
```

실제 커널은 FP16 half 로드 최적화 (`load_8_halves_ld128`) 와 `fmaf` 를 써서 wfma(warp) 수준으로 누산합니다.

---

## 6. Harmonic Encoding (주기적 활성화)

**파일**: `gsplat/gsplat/cuda/csrc/Interpolation.cuh`, `gsplat/gsplat/cuda/include/Common.h`

사면체 보간 결과 스칼라 `b` 에 대해 sin/cos 쌍을 적용합니다.

```cpp
// Common.h
#define NUM_ENCODING_FREQUENCIES 1
#define ENCF (NUM_ENCODING_FREQUENCIES * 2)   // = 2
#define FREQUENCY_SCALE_EXPONENTIAL 0          // 0=linear k+1, 1=exp 2^k

// Interpolation.cuh
get_encoding_frequency(k) = (k + 1)           // linear
                          or ldexpf(1, k)     // exponential 2^k

harmonic_encoding_fwd(b, f, out_sin, out_cos) {
    __sincosf(b * freq(f), out_sin, out_cos);
}
```

**물리적 해석**:

- `sin, cos` 은 서로 직교하는 Fourier 기저이므로, 서로 다른 주파수의 두 Gaussian 특징이 섞여도 **위상·진폭 정보를 분리된 채로 누적** 할 수 있습니다.
- `FREQUENCY_SCALE_EXPONENTIAL=1` 로 바꾸면 NeRF positional encoding 과 동일한 `2^k` 주파수 래더가 됩니다.
- alpha blending 결과가 본질적으로 Fourier 계수들의 가중 평균이므로, **렌더링 방정식 자체가 미분 가능한 주파수 믹서** 역할을 합니다.
- 고주파 텍스처(미세한 패턴, 광택 하이라이트)는 이 주파수 기저를 활용해 MLP 가 복원하므로, 3DGS가 놓치는 고주파 디테일을 채웁니다.

기본값 `NUM_ENCODING_FREQUENCIES=1` 은 "sin, cos 한 쌍" 만을 의미하며, 그 이상으로 늘리면 품질은 향상되지만 픽셀 텐서 폭이 선형 증가해 메모리/연산이 늘어납니다.

---

## 7. Deferred Shading MLP 디코더

**파일**: `gsplat/gsplat/nht/deferred_shader.py`

### 7.1 왜 Deferred 인가

기존 3DGS는 각 Gaussian 교차마다 **per-sample SH evaluation** 을 하기 때문에, Gaussian 수 N · 교차 수 M에 비례하는 SH 비용이 발생합니다. NHT는 조화 특징을 픽셀 텐서에 먼저 누적시킨 뒤, **픽셀당 딱 한 번만** MLP를 호출합니다.

```
3DGS SH 비용:     O(N · M · 48)    // per-primitive SH eval
NHT MLP 비용:     O(H · W · MLP)   // per-pixel only, primitive 수 무관
```

덕분에 primitive 수를 크게 늘려도 셰이딩 비용이 거의 늘지 않고, **고-primitive, 고-디테일 장면**에서 실시간 속도를 유지할 수 있습니다 (논문 Table 7).

### 7.2 구조

```python
DeferredShaderModule(
    feature_dim=48,           # F = 프리미티브당 특징 차원
    enable_view_encoding=True,
    view_encoding_type="sh",  # 또는 "fourier"
    sh_degree=3,              # view SH 차수
    sh_scale=3.0,             # 방향벡터 정규화 스케일
    mlp_hidden_dim=128,
    mlp_num_layers=3,         # 히든 레이어 3개 (ReLU) + Sigmoid 출력
)
```

tcnn의 `FullyFusedMLP` 로 구현되며 입력은 다음 `Composite` 인코딩으로 결합됩니다.

```
input = [
    Identity(encoded features, dim = (F/4) * ENCF),
    SphericalHarmonics(ray_dir, degree=3, dim = 9)  # 또는 Frequency 인코딩
]
out = Sigmoid(Linear(128) → ReLU → Linear(128) → ReLU → Linear(128) → ReLU → Linear(3))
```

### 7.3 Per-pixel vs Center-ray 뷰 인코딩

`--deferred_opt_center_ray_encoding` 옵션은 두 가지 모드를 선택합니다.

| 모드 | 설명 | 용도 |
|------|------|------|
| per-pixel (기본) | 픽셀마다 개별 광선 방향을 저장 | 왜곡 카메라, 고해상도 |
| center-ray | 타일 중앙 광선 1개만 MLP 입력으로 사용 | 속도 우선 (Table 1 indoor 설정) |

center-ray 모드는 타일 내 픽셀마다 같은 방향 SH 인코딩을 공유해 메모리를 절약합니다.

### 7.4 EMA

`--deferred_mlp_ema=True` (기본) 이면 학습 중 MLP 가중치의 지수 이동 평균(decay=0.95) 을 유지하고, 평가·렌더링 시에는 EMA 가중치를 사용합니다. 학습 초기 jitter 를 평활화해 PSNR 안정화에 도움을 줍니다.

---

## 8. 스크립트 실행 파이프라인

### 8.1 `scripts/train.sh`

**목적**: 단일 장면 NHT 학습.

```bash
bash scripts/train.sh                                   # garden, factor=4, cap=1M (기본)
bash scripts/train.sh --scene kitchen --data_factor 2
bash scripts/train.sh --scene bonsai --cap_max 2000000
```

핵심 명령 라인 (`scripts/train.sh:56-64`):

```bash
python gsplat/examples/simple_trainer_nht.py default \
    --data_dir   data/mipnerf360/garden \
    --data_factor 4 \
    --result_dir results/nht_mcmc_1000000/garden \
    --strategy.cap-max 1000000 \
    --render_traj_path ellipse
```

`simple_trainer_nht.py` 내부 흐름:

1. COLMAP 파서로 포즈·이미지·sparse point cloud 읽기.
2. `MCMCStrategy` 초기화 (NHT 특화 서브클래스: NaN-safe).
3. `HarmonicFeatures.init_features_random()` 로 `features [N, F]` 파라미터 생성.
4. `DeferredShaderModule` 구성 (tcnn FullyFusedMLP).
5. `rasterization(..., nht=True, with_eval3d=True, with_ut=True)` 호출.
6. Loss = L1 + λ·D-SSIM (+ LPIPS 평가 시).
7. Adam + SelectiveAdam(features) 로 역전파.
8. 일정 스텝마다 MCMC relocate/add 로 primitive 밀도 조절.

### 8.2 `scripts/eval.sh`

**목적**: 체크포인트 하나에 대해 품질 메트릭 + 런타임 타이밍.

```bash
bash scripts/eval.sh --ckpt results/.../ckpt_29999_rank0.pt
bash scripts/eval.sh --ckpt <ckpt> --skip_runtime
```

내부 흐름 (`scripts/eval.sh:93-172`):

1. 체크포인트를 로드해 `simple_trainer_nht.py` 를 eval-only 로 재실행 → `stats/val_step*.json` 에 PSNR/SSIM/LPIPS/#GS 기록.
2. 가장 최신 `val_step*.json` 을 파싱해 표준 출력.
3. `--skip_runtime` 가 아니면 `benchmarks/benchmark_nht.py` 호출:
   - CUDA events 로 **rasterization ms** 와 **deferred MLP ms** 를 분리 측정.
   - rasterizer → 전력 스파이크 → 다음 MLP 클록 저하 를 피하기 위해 **두 단계를 완전히 분리된 패스** 로 돌림.
   - 결과를 `stats/timing.json` 에 저장.

### 8.3 `scripts/view.sh`

**목적**: 체크포인트를 로드해 viser 인터랙티브 뷰어 실행.

```bash
bash scripts/view.sh --ckpt results/.../ckpt_29999_rank0.pt
# 브라우저: http://localhost:8080
```

`simple_viewer_nht.py` 에 `--ckpt`, `--output_dir`, `--port` 만 넘기면, 아래 render mode 를 UI 드롭다운에서 선택할 수 있습니다.

| Mode | 의미 |
|------|------|
| `rgb` | deferred MLP 가 디코딩한 최종 색상 |
| `depth(accumulated)` | 알파 가중 깊이 누적 |
| `depth(expected)` | 기대 깊이 = accumulated / alpha |
| `alpha` | 누적 불투명도 |

### 8.4 `benchmarks/nht/benchmark_nht.sh` (Table 2 재현)

```bash
bash benchmarks/nht/benchmark_nht.sh                      # 13 scenes
SCENE_LIST="garden bonsai truck" bash benchmarks/nht/benchmark_nht.sh
GPU=1 CAP_MAX=2000000 bash benchmarks/nht/benchmark_nht.sh
```

스크립트 로직:

1. 장면 목록을 MipNeRF360(실내/실외), Tanks&Temples, Deep Blending 4그룹으로 분류.
2. 각 장면의 `DATA_FACTOR` 를 자동 결정 (실내 2, 실외 4, TnT/DB 1).
3. `simple_trainer_nht.py mcmc` 를 **FEATURE_DIM=64, CAP_MAX=1M, MAX_STEPS=30k** 로 학습 → 체크포인트 생성.
4. 학습된 모든 `*.pt` 에 대해 eval 재실행 → 품질 저장.
5. 최종 결과 요약을 `$RESULT_BASE/*/stats/val_step*.json` 에서 수집해 출력.

다른 벤치마크 스크립트의 차이점:

| 스크립트 | Table | 특이점 |
|-----------|-------|--------|
| `benchmark_nht.sh` | Table 2 | 공정 비교용, 1M primitive 고정 |
| `benchmark_nht_split.sh` | Table 1 | 장면군별 best config (primitive 2M~5M) |
| `benchmark_nht_high.sh` | Table 7 | 장면당 별도 CAP_MAX로 품질 상한 추구 |
| `benchmark_nht_aov.sh` | - | LSEG/DINOv3/RGB2X 멀티헤드 (실험적) |

### 8.5 `benchmarks/benchmark_nht.py`

단독 실행형 런타임 벤치마크. 주요 옵션:

```bash
python benchmarks/benchmark_nht.py \
    --ckpt <path> --data_dir <scene> --data_factor 4 \
    --num_passes 3 --warmup_frames 10
```

- 각 장면을 **독립 subprocess** 로 실행해 GPU 상태 오염을 방지.
- `rasterization`, `deferred MLP`, `total` 을 **분리 패스**로 측정.
- `--collect_only` 를 주면 이전 실행의 `timing.json` 들만 모아 집계.
- 그룹 평균(M360-In/Out/T&T/DB/Overall)은 `aggregate_timing()` 에서 산술 평균으로 계산.

---

## 9. CUDA 래스터라이저 NHT 커널 분석

**파일**: `gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorldNHT3DGSFwd.cu`

### 9.1 커널 배치

```
gridDim  = (I,  tile_height,  tile_width)     // I = batch * camera
blockDim = (tile_size,  tile_size,  1)         // 기본 16×16 = 256 thread
shmem    = 256 · (sizeof(int32) + sizeof(vec4) + sizeof(mat3))
         ≈ 256 · (4 + 16 + 36) = 14,336 bytes
```

각 (이미지, 타일) 블록이 1 타일의 모든 픽셀을 협동하여 처리합니다.

### 9.2 주요 상수

| 상수 | 값 | 의미 |
|------|-----|------|
| `CDIM` | 템플릿 (예: 48) | 프리미티브당 특징 총 차원 |
| `VERTEX_PER_PRIM` | 4 | 사면체 꼭짓점 수 |
| `OUT_CDIM` | `CDIM/4` | 꼭짓점당 특징 길이 (12 when F=48) |
| `ENCF` | 2 | sin/cos 채널 쌍 |
| `FEAT_OUT` | `OUT_CDIM · ENCF` | 픽셀에 쓰이는 특징 수 (24 when F=48) |
| `PIXEL_STRIDE` | `FEAT_OUT + 3` | 픽셀당 출력 채널 (+ ray dir 3) |

### 9.3 광선 생성

각 픽셀은 자신의 pixel center `(j+0.5, i+0.5)` 로부터 **world-space ray** 를 생성합니다. 카메라 모델은 `camera_model_type` 에 따라 4가지 분기합니다.

```
CameraModelType::PINHOLE               (radial/tangential/thin_prism 지원)
CameraModelType::FISHEYE               (OpenCV fisheye 왜곡)
CameraModelType::FTHETA                (F-Theta 등거리)
CameraModelType::ORTHO                 (orthographic)
```

롤링 셔터 지원을 위해 `viewmats0`, `viewmats1` 두 시각의 ext-matrix 를 받아 `RollingShutterParameters` 로 시간 보간합니다.

### 9.4 Gaussian 교차 및 NHT 조회

광선을 Gaussian 로컬 좌표계로 이동시키는 과정이 핵심입니다 (`RasterizeToPixelsFromWorldNHT3DGSFwd.cu:213-220`).

```cpp
// 로컬 원점과 방향 (S · R^T 변환)
const vec3 gro = iscl_rot * (ray_o - xyz);         // S · R^T · (origin − μ)
const vec3 grd = safe_normalize(iscl_rot * ray_d); // 정규화된 로컬 방향

// 광선 위의 Gaussian 중심과 가장 가까운 점
const float t_closest = -dot(gro, grd);
const vec3 sample_pos_v = gro + t_closest * grd;
const float3 sample_pos = make_float3(sample_pos_v.x, ...);
const float grayDist = dot(sample_pos_v, sample_pos_v);

// Gaussian 값
const float power = -0.5f * grayDist;
float alpha = min(0.999f, opac * __expf(power));
if (alpha < 1/255) continue;
```

여기서 **`sample_pos` 가 NHT의 쿼리 좌표** 가 됩니다. 이 점에 대해 사면체 barycentric 가중치와 조화 인코딩이 수행됩니다.

조화 누적 루프의 FP16 최적화 경로 (`Fwd.cu:240-261`):

```cpp
for (k = 0; k < BASE_CDIM; k += 8) {
    // 4 꼭짓점 × 8 채널을 FP16 128bit load 로 1번에
    float acc[8] = {0};
    for (v = 0; v < 4; ++v) {
        float tmp[8];
        load_8_halves_ld128(f_base_ptr + v*BASE_CDIM + k, tmp);
        for (ii = 0; ii < 8; ++ii) acc[ii] = fmaf(weights[v], tmp[ii], acc[ii]);
    }
    // 조화 활성화 + 픽셀 누적 (vis = α · T)
    for (ii = 0; ii < 8; ++ii)
      for (freq = 0; freq < NUM_ENCODING_FREQUENCIES; ++freq) {
        float s, c;
        harmonic_encoding_fwd(acc[ii], freq, s, c);
        ACC_ADD(FREQ_IDX(k+ii, 2*freq),     s, vis);
        ACC_ADD(FREQ_IDX(k+ii, 2*freq + 1), c, vis);
      }
}
```

조기 종료 조건 `next_T ≤ 1e-4` 로 루프를 끊어 불필요한 계산을 피합니다.

### 9.5 픽셀 출력 레이아웃

```
render_colors[pix][0]                     ┐
                                          │  FEAT_OUT 채널
render_colors[pix][FEAT_OUT − 1]          ┘

render_colors[pix][FEAT_OUT + 0] = (dx · scale + 1) · 0.5  ─┐
render_colors[pix][FEAT_OUT + 1] = (dy · scale + 1) · 0.5   │  tcnn [0,1] 매핑
render_colors[pix][FEAT_OUT + 2] = (dz · scale + 1) · 0.5  ─┘
```

여기서 ray 방향은 tcnn 의 `SphericalHarmonics` 인코딩이 기대하는 `[0, 1]` 범위로 선형 매핑됩니다. `ray_dir_scale` 은 `DeferredShaderModule.ray_dir_scale` (기본 `sh_scale=3.0`) 로 조절됩니다.

`render_alphas[pix] = 1 − T` 는 누적 불투명도이며, 배경 합성이나 뷰어의 alpha map 모드에 사용됩니다.

### 9.6 공유메모리 프리페치

각 iteration 에서 256 스레드가 256 Gaussian 의 다음 데이터를 동시에 SMEM 으로 올립니다.

```cpp
xyz_opacity_batch[tr] = {xyz.x, xyz.y, xyz.z, opac};
iscl_rot_batch[tr]    = S · R^T;   // Gaussian 로컬 변환
```

이후 256 픽셀-스레드가 공유된 batch 를 순차 순회합니다. 이는 **L2/글로벌 메모리 접근 수를 1/256 으로 줄이는** 고전적 splat 최적화입니다.

---

## 10. 역전파 (Backward Pass)

**파일**: `gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorldNHT3DGSBwd.cu`

NHT backward 는 3DGS와 동일하게 **픽셀의 Gaussian hit 을 역순** 으로 순회하면서 알파 합성 chain rule 을 적용합니다. 달라지는 지점은 "색상" 이 아니라 **사면체 특징과 조화 기저** 에 대한 기울기를 계산한다는 점입니다.

```
dL/d(alpha), dL/d(T)  (알파 합성 역산)
     │
     ├─ dL/d(h_sin, h_cos)   // 픽셀 누적 기울기
     │
     │    # harmonic_encoding_bwd
     │    ∂sin(f·b)/∂b =  f · cos(f·b)
     │    ∂cos(f·b)/∂b = −f · sin(f·b)
     │
     ├─ dL/d(b_k)            // per-channel interpolated feature
     │
     │    # barycentric_interpolate_bwd
     │    dL/d(v_i[k]) = w_i · dL/d(b_k)
     │    dL/d(w_i)    = Σ_k v_i[k] · dL/d(b_k)
     │    dL/d(sample_pos) = Σ_i dL/d(w_i) · n_i · TET_INV_H
     │
     ├─ dL/d(features[v, k])  // 프리미티브 특징 기울기 (SelectiveAdam 에서 업데이트)
     └─ dL/d(μ, R, S, α)      // 위치/회전/스케일/불투명도 기울기
```

구현 포인트:

- backward 는 **FP32 누산** 을 고집합니다 (features 는 FP16 로드하더라도 grad 는 FP32).
- feature 기울기는 다른 thread 와 충돌하므로 `atomicAdd` 로 반영하되, 먼저 warp 수준 감축으로 lane 0 만 atomic 을 호출해 경합을 줄입니다.
- `sample_pos` 기울기는 Gaussian 중심(μ), 회전(R), 스케일(S) 의 기울기로 chain 됩니다. `iscl_rot = S · R^T` 의 미분을 명시적으로 계산해 `v_means`, `v_quats`, `v_scales` 에 누적합니다.
- `strategy.py` 의 `_sanitize_opacities` 는 학습 초반 NaN opacity 가 들어오면 0 으로 덮어, multinomial sampling 실패를 막습니다.

---

## 11. MCMC 밀도화 전략

**파일**: `gsplat/gsplat/nht/strategy.py`

기본 전략은 `gsplat` 의 `MCMCStrategy` 를 상속하며, NHT 관점에서 **opacity logit 을 정상화** 하는 safety layer 만 추가합니다. MCMC 의 상위 흐름은 다음과 같습니다.

```
매 N step 마다:
  ① 현재 opacities → multinomial sampling 으로 "살아남을" 파티클 선택
  ② 생존 파티클을 scale 절반으로 쪼개거나 (split) 같은 자리에 복제 (clone)
  ③ dead 파티클을 reset 하고 새로운 위치에 re-seed
  ④ primitive 수가 cap_max 이하가 되도록 조정
```

NHT에서는 `cap_max` 가 밀도화의 핵심 상한이 되어, 예컨대 `--strategy.cap-max 1000000` 로 1M 프리미티브에서 학습이 수렴하도록 유도됩니다. 이 값이 커질수록 품질은 오르지만 VRAM 과 학습 시간이 증가합니다.

`NHTMCMCStrategy._sanitize_opacities()` 는 매 relocate/add 호출 전에 NaN 을 0 으로 치환합니다. NHT 는 초기 에포크에 조화 신호가 크게 튈 수 있어 opacity logit 이 발산하기 쉬워, 이 패치가 학습 안정성을 크게 향상시킵니다.

---

## 12. AOV 확장 (멀티헤드 추론)

**파일**: `aov/deferred_shader.py`, `aov/examples/simple_trainer_nht_aov.py`

### 12.1 개념

AOV (Arbitrary Output Variables) 는 RGB 외에 **의미 분할 피처(LSEG, DINOv3)** 또는 **PBR 재질(RGB2X: albedo, roughness, metallic ...)** 을 함께 회귀하는 멀티헤드 모드입니다. 학습은 **사전 계산된** feature map 을 supervision 으로 사용합니다.

### 12.2 아키텍처 선택

`DeferredShaderAOVModule` 은 보조 출력 차원 `K = LSEG + DINOv3 + RGB2X` 에 따라 4가지 아키텍처를 자동 선택합니다 (`aov/deferred_shader.py:45-123`).

| 조건 | 아키텍처 | 설명 |
|------|----------|------|
| `K == 0` | `rgb_only_sigmoid` | 기존 NHT 와 동일 |
| semantic >0 | `split_rgb_aux_linear` | tcnn(128) → RGB 3 + Linear(125→K) |
| rgb2x only & `3+K < 128` | `fused_direct` | tcnn 이 `3+K` 를 한 번에 sigmoid |
| 대용량 | `full_linear_readout` | tcnn(128) → Linear(128 → 3+K) |

tcnn 이 이미 Sigmoid 를 적용하는 경우 (`tcnn_emitted_sigmoid_outputs == True`), 래퍼는 이중 squash 를 피하기 위해 identity 활성화로 fall-through 됩니다.

### 12.3 Forward 출력

```python
colors, aov_outputs, extras = shader(rendered_data)
# aov_outputs = {"lseg": [...], "dinov3": [...], "rgb2x_albedo": [...], ...}
```

RGB 는 sigmoid, LSEG/DINOv3 는 linear (코사인 유사도 학습), RGB2X 는 sigmoid 가 기본입니다.

> AOV는 **실험적** 이며, 품질/성능이 불안정할 수 있고 사전 학습된 DINOv3/LSEG/RGB2X 모델은 레포에 포함되지 않습니다. 별도의 전처리 파이프라인이 필요합니다.

---

## 13. 하이퍼파라미터 설명

### 13.1 NHT 고유 파라미터

| 파라미터 | 기본값 | 설명 / 물리적 의미 |
|----------|--------|---------------------|
| `--deferred_opt_feature_dim` | 48 | 프리미티브당 총 특징 차원 F. 4로 나뉘어 각 사면체 꼭짓점에 분배 |
| `--deferred_features_lr` | 0.015 | 특징 학습률 (SelectiveAdam 사용) |
| `--deferred_mlp_lr` | 0.00068 | Deferred MLP 학습률 (features 보다 훨씬 작음) |
| `--deferred_mlp_hidden_dim` | 128 | MLP 히든 너비. tcnn FullyFusedMLP 제약으로 32/64/128 권장 |
| `--deferred_mlp_num_layers` | 3 | 히든 레이어 수. 깊을수록 표현력↑, 오버헤드↑ |
| `--deferred_mlp_ema` | True | EMA(0.95) 로 MLP 안정화 |
| `--deferred_opt_center_ray_encoding` | False | 타일 중심 광선 공유 모드 |
| `--deferred_opt_view_encoding_type` | `"sh"` | 시점 인코딩: `sh` (저차원) 또는 `fourier` (고주파) |
| `--deferred_opt_sh_degree` | 3 | SH 차수 (9 채널) |
| `--deferred_opt_sh_scale` | 3.0 | 방향벡터 사전 스케일 (tcnn [0,1] 매핑 전) |
| `--deferred_lr_scheduler` | `"cosine"` | cosine 또는 exponential LR 스케줄 |
| `--color_refine_steps` | 3000 | 학습 마지막 N step 은 geometry freeze, 색상만 세밀 조정 |

### 13.2 밀도화 / 정규화

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--strategy.cap-max` | 1,000,000 | 유지되는 프리미티브 수 상한 |
| `--opacity_reg` | 0.02 | opacity L1 정규화 (뚜렷한 on/off 유도) |
| `--scale_reg` | 0.005 | scale 로그 L1 정규화 (크기 폭주 방지) |

### 13.3 렌더 / 로스

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--ssim_lambda` | 0.1 | D-SSIM 가중치 (총 loss 중 10%) |
| `--tile_size` | 16 | 타일 변 길이. `F` 가 크면 8 로 낮춰 SMEM 절약 |
| `--lpips_net` | `"vgg"` | LPIPS 백본 |
| `--lpips_normalize` | True | VGG 입력을 `[-1,1]` 로 정규화 (NHT 논문 기본, INRIA와 다름) |

---

## 14. 참고 자료

- **논문 PDF**: [Neural Harmonic Textures (research.nvidia.com)](https://research.nvidia.com/labs/sil/projects/neural-harmonic-textures/assets/neural_harmonic_textures.pdf)
- **프로젝트 페이지**: [research.nvidia.com/labs/sil/projects/neural-harmonic-textures](https://research.nvidia.com/labs/sil/projects/neural-harmonic-textures/)
- **원본 코드**: [github.com/nv-tlabs/neural-harmonic-textures](https://github.com/nv-tlabs/neural-harmonic-textures)
- **gsplat 기반 구현 서브모듈**: [github.com/nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) (branch: `nv/nht-initial-release`)
- **관련 작업**: 3DGS [Kerbl et al. 2023], 3DGUT [Wu et al. 2025], Instant-NGP [Müller et al. 2022]

### 핵심 코드 참조

| 주제 | 파일 경로 |
|------|-----------|
| 사면체 barycentric + 조화 인코딩 | `gsplat/gsplat/cuda/csrc/Interpolation.cuh` |
| NHT forward 래스터라이즈 | `gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorldNHT3DGSFwd.cu` |
| NHT backward 래스터라이즈 | `gsplat/gsplat/cuda/csrc/RasterizeToPixelsFromWorldNHT3DGSBwd.cu` |
| Deferred MLP 모듈 | `gsplat/gsplat/nht/deferred_shader.py` |
| MCMC 밀도화 전략 | `gsplat/gsplat/nht/strategy.py` |
| 학습 루프 | `gsplat/examples/simple_trainer_nht.py` |
| 벤치마크 러너 | `benchmarks/benchmark_nht.py` |
| Table 2 스크립트 | `benchmarks/nht/benchmark_nht.sh` |

---

*한국어 번역 및 코드 해설: [leonyoon-3dai fork](https://github.com/leonyoon-3dai/neural-harmonic-textures). 오류/개선 제안은 이슈로 부탁드립니다.*
