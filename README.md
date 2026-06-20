# UI Usability Evaluation

Rico 모바일 UI 스크린샷과 계층 정보를 사용해 화면을 `usable` 또는 `risk`로 분류하는 실험 프로젝트입니다.
이미지만 사용하는 CNN 계열 모델과, 이미지 특징에 레이아웃 특징을 함께 쓰는 `ui_aware_hybrid` 모델을 같은 조건에서 비교합니다.

## 데이터 준비

실험을 실행하려면 아래 경로에 Rico 데이터가 있어야 합니다.

```text
data/screenshots/
data/hierarchies/
```

원본 Rico 데이터는 용량이 커서 저장소에 포함하지 않습니다.
라벨은 실제 사용자 평가가 아니라, 화면 이미지와 계층 정보에서 관찰 가능한 정적 규칙으로 만든 약한 라벨입니다.

## 모델

- `cnn`: 기본 CNN 모델
- `resnet18`: ResNet18 기반 이미지 분류 모델
- `efficientnet_b0`: EfficientNet-B0 기반 이미지 분류 모델
- `ui_aware_hybrid`: 스크린샷 특징과 레이아웃 특징을 함께 사용하는 모델

## 실행 방법

필요한 패키지를 설치합니다.

```powershell
py -3.12 -m pip install -r requirements.txt
```

빠른 동작 확인은 smoke 설정으로 실행합니다.

```powershell
py -3.12 src\main.py --mode w3c-cv --config configs\w3c_rule_cv_experiment.yaml
```

전체 모델 비교 실험은 full 설정으로 실행합니다.

```powershell
py -3.12 src\main.py --mode w3c-cv --config configs\w3c_rule_cv_experiment_full.yaml
```

## 산출물

실행 결과는 `outputs` 아래에 생성됩니다.
이 폴더는 로컬 실행 결과이므로 Git에는 포함하지 않습니다.

```text
outputs/runs/        학습 및 평가 결과
outputs/cache/       이미지 텐서 캐시
outputs/standards/   웹 접근성 기준 추출 결과
outputs/labels/      약한 라벨 생성 로그
```

주요 결과 파일은 보통 다음과 같습니다.

```text
metrics.json
training_history.json
split.json
plots/
```

## 주의

이 프로젝트의 라벨은 실제 사용자 만족도나 과업 성공률이 아닙니다.
따라서 결과는 모델 간 비교와 경향 분석에는 사용할 수 있지만, 실제 사용성 평가나 접근성 적합성 판정을 대신하지 않습니다.
