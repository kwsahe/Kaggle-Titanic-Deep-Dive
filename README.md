# 빅데이터분석기사 실기 연습 노트

빅데이터분석기사 실기 시험을 준비하면서 만든 개인 연습용 프로젝트입니다.  
Titanic 데이터셋을 활용한 전처리/모델링 연습과, 6~9회 기출 유형별 풀이 자료를 함께 정리했습니다.

## 학습 범위

- 1유형: Pandas 기반 데이터 전처리, 필터링, 집계, 결측치/이상치 처리
- 2유형: 머신러닝 모델 학습, 검증, 예측 결과 제출 파일 생성
- 3유형: 통계 검정, 회귀분석, p-value 해석, ANOVA, 카이제곱 검정
- 시험 직전 복습용 코드 조각과 개념 메모 정리

## 프로젝트 구조

```text
.
|-- notebooks/
|   |-- titanic_practice.ipynb
|   `-- exam_practice/
|       |-- test_code.ipynb
|       `-- test_code_final.ipynb
|-- notes/
|   |-- memorize_code.txt
|   |-- type1_code.md
|   |-- type2_code.md
|   |-- type3_code.md
|   |-- type3_concept.md
|   `-- type_all_code_1125.md
|-- past_questions/
|   |-- round_06/
|   |-- round_07/
|   |-- round_08/
|   `-- round_09/
|-- outputs/
|   |-- titanic_result.csv
|   |-- practice_result.csv
|   |-- practice_result1.csv
|   `-- practice_result2.csv
`-- README.md
```

## 폴더 설명

| 경로 | 설명 |
| --- | --- |
| `notebooks/titanic_practice.ipynb` | Titanic 데이터셋으로 전처리, 파생변수 생성, 모델링, 통계 검정을 연습한 노트북 |
| `notebooks/exam_practice/` | 실기 유형별 종합 연습 노트북 |
| `notes/` | 시험 직전 확인용 코드 스니펫, 유형별 풀이 패턴, 통계 개념 정리 |
| `past_questions/round_06`~`round_09` | 회차별 기출 풀이 노트북과 해당 CSV 데이터 |
| `outputs/` | 모델 예측 결과 또는 제출 형식으로 저장한 CSV 파일 |

## 사용 방법

1. Jupyter Notebook 또는 VS Code에서 원하는 노트북을 엽니다.
2. 기출 회차별 풀이를 볼 때는 `past_questions/round_XX` 폴더 안의 `*_code.ipynb`를 실행합니다.
3. 유형별 문법이나 자주 쓰는 코드는 `notes/` 폴더에서 확인합니다.
4. 예측 결과 파일은 `outputs/` 폴더에서 확인합니다.

## 주요 라이브러리

노트북은 주로 아래 라이브러리를 사용합니다.

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

from scipy import stats
from statsmodels.formula.api import ols, logit, glm
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
```

## 정리 기준

- 학습 노트북은 `notebooks/`로 분리했습니다.
- 회차별 기출 자료는 노트북과 CSV가 함께 있어야 상대경로 실행이 편하므로 `past_questions/round_XX/` 단위로 묶었습니다.
- 제출 결과물은 `outputs/`에 따로 모았습니다.
- 시험 직전 복습 자료는 `notes/`에 모았습니다.

## 참고

일부 노트북과 CSV는 연습 당시 작성한 원본 상태를 유지했습니다.  
따라서 환경에 따라 한글 주석이나 CSV 헤더 인코딩이 다르게 보일 수 있습니다.
