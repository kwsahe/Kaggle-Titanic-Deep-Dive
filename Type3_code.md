## 빅데이터분석기사 3유형 (통계 검정) 핵심 요약 ⚡️

### 1\. 📦 필수 라이브러리 (먼저 복붙\!)

시험 시작 시, 이 코드 블록을 먼저 실행하면 편리합니다.

```python
import pandas as pd
import numpy as np

# 1. t검정, 카이제곱, 정규성, 상관분석 등
from scipy import stats

# 2. 비율 검정
from statsmodels.stats.proportion import proportions_ztest

# 3. 회귀분석 (OLS, Logit)
from statsmodels.formula.api import ols, logit, glm
import statsmodels.api as sm # GLM의 family 지정 시 필요

# 4. 분산분석 (ANOVA)
from statsmodels.stats.anova import anova_lm

# 5. 사후검정 (Tukey)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
```

-----

### 2\. ⚖️ p-value 판결의 모든 것 (이것만 기억\!)

모든 검정의 결론은 `p-value`가 0.05보다 작은지만 보면 됩니다.

  * **`p-value <= 0.05` (5%보다 작다)**
      * "이 차이는 우연이 아니다\! (유의미하다)"
      * **판결: 귀무가설($H_0$) 기각** (➡️ 대립가설($H_1$) 채택)
  * **`p-value > 0.05` (5%보다 크다)**
      * "이 차이는 우연일 수 있다. (유의미하지 않다)"
      * **판결: 귀무가설($H_0$) 기각 실패** (그대로 유지)

-----

### 3\. 📋 실전\! '문제 유형별' 코드 족보

내가 풀어야 할 문제가 무엇인지 확인하고, 해당 코드를 찾아 쓰세요.

#### 1️⃣ "평균" 비교 (수치형 데이터 📊)

**(사전검사) 정규성 검정 (Shapiro-Wilk)**

  * 데이터가 정규분포를 따르는지 확인합니다.
  * `H0`: 정규분포를 따른다.
  * `p > 0.05` 여야 정규분포를 만족하여 t-검정/ANOVA를 쓸 수 있습니다.

<!-- end list -->

```python
# stat, p = stats.shapiro(data)
```

**(A) 2개 그룹 평균 비교 (t-검정)**

  * `H0`: 두 그룹의 평균은 같다.

<!-- end list -->

```python
# 1. 독립표본 (예: A반 vs B반)
# (등분산 가정: equal_var=True)
stats.ttest_ind(group1_score, group2_score, equal_var=True)

# 2. 대응표본 (예: 복용 전 vs 복용 후)
stats.ttest_rel(before_score, after_score)

# 3. 단일표본 (예: 우리 반 vs 전국 평균)
stats.ttest_1samp(sample_scores, popmean=75)
```

**(B) 3개 이상 그룹 평균 비교 (분산분석 - ANOVA)**

  * `H0`: 모든 그룹의 평균은 같다.

<!-- end list -->

```python
# 1. ANOVA 실행 (Scipy 방식 - 가장 간단)
stats.f_oneway(group1_score, group2_score, group3_score)

# 2. ANOVA 실행 (Statsmodels 방식 - F값, P값 모두 제공)
model = ols('score ~ C(group_col)', data=df).fit()
result = anova_lm(model)
# print(result) # p-value는 PR(>F) 컬럼 확인

# 3. (필수) 사후검정 (ANOVA가 H0 기각 시, '누가' 다른지 확인)
posthoc = pairwise_tukeyhsd(df['score_col'], df['group_col'])
# print(posthoc) # reject=True인 그룹이 유의미한 차이
```

#### 2️⃣ "빈도수/비율" 비교 (범주형 데이터 🧮)

**(A) 두 변수 간 "관련성" (카이제곱 - 독립성 검정)**

  * `H0`: 두 변수는 서로 관련이 없다 (독립이다).

<!-- end list -->

```python
# 1. (필수) 교차표(Crosstab) 생성
ct = pd.crosstab(df['category_A'], df['category_B'])

# 2. 카이제곱 검정 실행
chi2, p, dof, expected = stats.chi2_contingency(ct)
```

**(B) "예상과 일치" (카이제곱 - 적합도 검정)**

  * `H0`: 실제 관측 빈도가 기대 빈도와 같다.

<!-- end list -->

```python
# f_obs: 실제 관측 빈도 리스트 (예: [30, 25, 45])
# f_exp: 기대 빈도 리스트 (예: [33, 33, 34])
stats.chisquare(f_obs=observed, f_exp=expected)
```

**(C) 두 집단 "비율" 차이 (비율 검정)**

  * `H0`: 두 집단의 비율은 같다.

<!-- end list -->

```python
# 예: A집단 100명 중 30명 성공, B집단 100명 중 40명 성공
stat, p = proportions_ztest(count=[30, 40], nobs=[100, 100])
```

#### 3️⃣ "예측 모델링" (회귀 분석 🎯)

**(A) "숫자" 예측 (선형 회귀 - OLS)**

  * `H0`: 해당 변수(x)는 y에 영향을 주지 않는다.

<!-- end list -->

```python
# R-squared(결정계수)와 각 변수의 P>|t|(p-value)를 확인
model = ols('y ~ x1 + x2', data=df).fit()
# print(model.summary())
```

**(B) "범주" 예측 (로지스틱 회귀 - Logit / GLM)**

  * `H0`: 해당 변수(x)는 target에 영향을 주지 않는다.

<!-- end list -->

```python
# 1. Logit (주로 사용)
model = logit('target ~ x1 + x2', data=df).fit()
# print(model.summary())

# 2. GLM (Logit과 결과 동일)
family = sm.families.Binomial()
model = glm('target ~ x1 + x2', data=df, family=family).fit()
# print(model.summary())
```

-----

### 4\. ⚡️ 실전 예제 코드 (복붙용 템플릿)

#### 📈 t-검정 (독립표본)

```python
from scipy import stats

# 1. 그룹 분리 (예: group 컬럼이 'A'인 데이터의 'score'만 추출)
groupA = df[df['group'] == 'A']['score']
groupB = df[df['group'] == 'B']['score']

# 2. t-검정 (등분산 가정)
stat, p = stats.ttest_ind(groupA, groupB, equal_var=True)
print(f'p-value: {p:.3f}')
```

#### 📊 분산분석 (ANOVA) + 사후검정

```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. ANOVA 모델 (y ~ C(X), C()는 group이 범주형임을 명시)
model = ols('score ~ C(method)', data=df).fit()
result = anova_lm(model)
print(result) # PR(>F) 컬럼의 p-value 확인

# 2. 사후검정 (p-value < 0.05 일 때만 실행)
posthoc = pairwise_tukeyhsd(df['score'], df['method'])
print(posthoc) # reject=True 확인
```

#### 🧮 카이제곱 검정 (독립성)

```python
from scipy.stats import chi2_contingency
import pandas as pd

# 1. 교차표 작성
ct = pd.crosstab(df['성별'], df['만족도'])

# 2. 검정 실행
chi2, p, dof, expected = chi2_contingency(ct)
print(f'p-value: {p:.3f}')
```

#### 🎯 로지스틱 회귀

```python
from statsmodels.formula.api import logit

# 1. 모델 피팅 (target이 0 또는 1이어야 함)
model = logit('target ~ age + income', data=df).fit()

# 2. 결과 요약 (P>|z| 컬럼의 p-value 확인)
print(model.summary())
```