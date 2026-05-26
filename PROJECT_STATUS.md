# CG-DARTS 프로젝트 진행 현황

제안서(CG_DARTS_Proposal_v3) 기준으로 DARTS CNN search에 **MAC/FLOPs 비용 정규화**를 architecture loss에 추가한 구현입니다.

```
L_arch = L_val + λ · E[MAC(α)]
```

## 완료된 항목

| 영역 | 상태 | 설명 |
|------|------|------|
| 핵심 구현 | ✅ | `cost_utils.py`, `architect_cg.py`, `train_search_cg.py` |
| 비용 메트릭 | ✅ | FLOPs / params, edge/global/none 정규화, λ warmup |
| 실행 스크립트 | ✅ | `run.sh`, `run_cg_sweep.sh` |
| 리포트 스크립트 | ✅ | `scripts/cg_darts_report.py` (실험 디렉터리 필요) |
| 이전 실험 요약 | ✅ | `reports/cg_darts/summary.csv` (2026-05-17~19, seed=2) |

### 이전 실험 결과 요약 (`summary.csv`)

| 설정 | Search valid acc | Expected cost ↓ | Discrete MACs ↓ | Retrain test acc |
|------|------------------|-----------------|-----------------|------------------|
| Vanilla | 87.93% | — | — | **95.97%** |
| λ=1e-3 | 87.52% | 5.4% | -6.6% (증가) | ❌ 미실행 |
| λ=5e-3 | 87.28% | 9.7% | 10.8% | ❌ 미실행 |
| λ=1e-2 | 87.61% | 16.9% | 12.1% | **96.05%** |
| λ=5e-2 | 87.10% | 58.1% | 46.5% | 93.96% |
| λ=1e-1 | 86.77% | 76.6% | 40.8% | ❌ 미실행 |

> **λ=1e-2**가 accuracy–cost 균형이 가장 좋음 (retrain 96.05%, MACs 12% 절감).

## 미완료 / 블로커

1. **실험 아티팩트 미동기화** — `search-*`, `eval-*` 디렉터리는 `.gitignore` 대상이라 이 워크스페이스에 없음. `cg_darts_report.py`가 실패하는 원인.
2. **Retrain 3건 미실행** — λ ∈ {1e-3, 5e-3, 1e-1} (genotype.txt 필요).
3. **리포트 플롯** — `summary.csv`만 있고 PNG 미생성.
4. **다중 seed / second-order** — 현재 seed=2, first-order만 수행된 것으로 보임.
5. **최종 600 epoch 평가** — DARTS 논문 full train 미실행.
6. **params 메트릭 sweep** — flops만 sweep됨.

## 권장 실행 순서 (GPU)

```bash
cd /home/members/ryeowook/cg-darts
export PYTHON_BIN=/home/members/ryeowook/miniconda3/bin/python

# FLOPs-cost sweep (기존)
LAMBDAS="1e-3 5e-3 1e-2 5e-2 1e-1" METRIC=flops ./run_cg_sweep.sh

# Parameter-count cost sweep (fast mode: 30ep search, 50ep retrain, λ∈{1e-2,5e-2}, 2 GPU)
./scripts/run_params_pipeline.sh
tail -f logs/params_search_gpu0.log logs/params_search_gpu1.log
# Override: EPOCHS=30 RETRAIN_EPOCHS=50 LAMBDAS="1e-2 5e-2" ./scripts/run_params_pipeline.sh

# 리포트
$PYTHON_BIN scripts/cg_darts_report.py --cost-metric params
```

## 환경

- GPU: NVIDIA L40S × 2 (`CUDA_VISIBLE_DEVICES`로 선택)
- Python: `/home/members/ryeowook/miniconda3/bin/python`
- CIFAR-10: `cg-darts/data/` (`--download`로 자동 다운로드)
