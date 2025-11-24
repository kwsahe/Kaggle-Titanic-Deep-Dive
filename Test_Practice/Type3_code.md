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

네, 빅데이터분석기사 실기 시험(특히 2, 3유형)에서 자주 쓰는 `import` 구문과 함수 반환값(`stat, p_val ...`)을 암기하기 쉽게 총정리했습니다.

이것만 외워두시면 코딩 시간을 획기적으로 줄일 수 있습니다.

-----

## 📦 1. `import` 총정리 (유형별 암기)

### 1\. 기본 (1, 2, 3유형 공통)

데이터를 불러오고, 다루고, 계산하는 데 필수입니다.

```python
# 판다스 (데이터프레임 핸들링)
import pandas as pd

# 넘파이 (수학 계산, 배열)
import numpy as np
```

### 2\. 머신러닝 (2유형 - 모델링)

`sklearn` (Scikit-Learn)이 핵심입니다.

```python
# 1. 데이터 분리 (필수)
from sklearn.model_selection import train_test_split

# 2. 전처리 (인코딩)
from sklearn.preprocessing import LabelEncoder # (예: 'male' -> 0)
from sklearn.preprocessing import StandardScaler # (숫자 스케일링)
from sklearn.preprocessing import MinMaxScaler # (숫자 스케일링)

# 3. 모델 선택 (2개만 외워도 충분)
from sklearn.ensemble import RandomForestClassifier # (분류 - 성능 좋음)
from sklearn.linear_model import LogisticRegression # (분류 - 기본)
# from sklearn.ensemble import RandomForestRegressor # (회귀 - 숫자 예측)
# from sklearn.linear_model import LinearRegression # (회귀 - 기본)

# 4. 평가지표 (문제에서 요구하는 것)
from sklearn.metrics import accuracy_score # (정확도)
from sklearn.metrics import f1_score # (F1 스코어)
from sklearn.metrics import roc_auc_score # (ROC AUC)
# from sklearn.metrics import mean_squared_error # (MSE - 회귀)
```

### 3\. 통계 검정 (3유형 - 가설 검정)

`scipy.stats`와 `statsmodels`가 핵심입니다.

```python
# 1. Scipy (단일 함수들)
from scipy import stats
# (여기에 ttest_ind, f_oneway, levene, chi2_contingency, 
#  chisquare, pearsonr, shapiro, wilcoxon 등이 모두 들어있음)

# 2. Statsmodels (회귀분석, ANOVA 상세 분석)
import statsmodels.api as sm # (GLM 등에서 family 지정 시 필요)
from statsmodels.formula.api import ols, logit, glm # (회귀식)
from statsmodels.stats.anova import anova_lm # (분산분석표)
from statsmodels.stats.multicomp import pairwise_tukeyhsd # (투키 사후검정)
from statsmodels.stats.proportion import proportions_ztest # (비율 검정)
```

-----

## 🐍 2. 함수 반환값 총정리 (stat, p\_val...)

`scipy.stats`의 많은 함수는 결과를 \*\*튜플(tuple)\*\*로 반환합니다. 이 값들을 순서대로 받아오는 것입니다.

### 1\. (통계량, p값) 2개를 반환하는 함수들

가장 일반적인 형태입니다. **거의 모든 검정**이 이 형식을 따릅니다.

  * `statistic`: 검정통계량 (t-값, F-값, $\chi^2$-값, 상관계수 r 등)
  * `pvalue`: p-값

<!-- end list -->

```python
# 템플릿: stat, p_val = function(...)

# 1. t-검정 (3종 세트)
stat, p_val = stats.ttest_1samp(sample, popmean=0)
stat, p_val = stats.ttest_ind(group1, group2, equal_var=True)
stat, p_val = stats.ttest_rel(before, after)

# 2. ANOVA (scipy 방식)
stat, p_val = stats.f_oneway(group1, group2, group3)

# 3. 정규성 검정 (Shapiro-Wilk)
stat, p_val = stats.shapiro(data)

# 4. 상관분석 (Pearson)
# (주의: 첫 번째 stat이 '상관계수 r'임)
corr, p_val = stats.pearsonr(x, y)

# 5. 적합도 검정 (Chisquare)
stat, p_val = stats.chisquare(f_obs=[...], f_exp=[...])

# 6. 등분산 검정 (Levene)
stat, p_val = stats.levene(group1, group2)

# 7. 비율 검정 (Statsmodels)
stat, p_val = proportions_ztest(count=[...], nobs=[...])

# 8. 부호 검정 (wilcoxon)
stat, p_val = wilcoxon(pre, post, alternative='two-sided' or 'greater' or 'less')
```

### 2\. (통계량, p값, 자유도, 기대빈도) 4개를 반환하는 함수

**`chi2_contingency` (카이제곱 독립성 검정)가 유일**합니다. 이것만 따로 외우세요.

```python
# 템플릿: chi2, p_val, dof, expected = stats.chi2_contingency(crosstab)

# 1. 교차표 생성
ct = pd.crosstab(df['Var1'], df['Var2'])

# 2. 검정 수행
chi2_stat, p_val, dof, expected = stats.chi2_contingency(ct)
```

  * `chi2_stat`: 카이제곱 통계량
  * `p_val`: p-값
  * `dof`: 자유도 (Degrees of Freedom)
  * `expected`: 기대빈도표 (Numpy 배열)

### 3\. (기타) 테이블/객체를 반환하는 함수들

`stat, p_val` 형식이 아닌 함수들입니다.

```python
# 1. ANOVA (Statsmodels 방식)
# 'DataFrame'을 반환
model = ols('y ~ C(X)', data=df).fit()
anova_table = anova_lm(model)
# print(anova_table) # -> PR(>F) 컬럼이 p-value

# 2. Tukey 사후검정
# 'TukeyHSDResult'라는 전용 객체(표)를 반환
tukey_result = pairwise_tukeyhsd(endog=df['value'], groups=df['group'])
# print(tukey_result) # -> p-adj 컬럼이 p-value
```

로지스틱 회귀분석은 \*\*"결과를 해석하는 능력"\*\*이 3유형의 핵심입니다. 요청하신 내용(Summary 해석, 오즈비, 특정 조건 계산)을 시험 실전용으로 딱 정리해 드릴게요.

-----

## 1\. `model.summary()` 완벽 해부 (시험에 볼 건 딱 2개\!)

`print(model.summary())`를 하면 복잡한 표가 나오지만, 시험에서는 \*\*`coef`\*\*와 \*\*`P>|z|`\*\*만 보면 됩니다.

### ① `coef` (회귀계수, Coefficient)

  * **의미:** 이 변수가 1 증가할 때, \*\*"로그 오즈(Log Odds)"\*\*가 얼마나 변하는지를 나타냅니다.
  * **해석:**
      * **양수(+)**: 이 변수가 커지면 성공(Target=1) 확률이 **높아진다.**
      * **음수(-)**: 이 변수가 커지면 성공(Target=1) 확률이 **낮아진다.**
  * **주의:** 이 값 자체는 확률이 아닙니다\! \*\*지수함수($e$)를 씌워야 "오즈비"\*\*가 됩니다.

### ② `P>|z|` (p-value, 유의확률)

  * **의미:** 이 변수가 통계적으로 유의미한가?
  * **기준:** **0.05보다 작아야 유의미**합니다. (0.05보다 크면 이 변수는 결과에 영향을 주지 않는다고 봅니다.)

-----

## 2\. 오즈비 (Odds Ratio)란?

시험 문제의 단골 손님입니다.

  * **오즈(Odds):** $\frac{\text{성공 확률}}{\text{실패 확률}}$
  * **오즈비(Odds Ratio):** 변수가 1 증가할 때 **오즈가 몇 배가 되는가?**
  * **공식:** $\text{Odds Ratio} = e^{\text{coef}} = \exp(\text{coef})$

> **시험 꿀팁:** 문제에서 "오즈비를 구하시오"라고 하면 무조건 \*\*`np.exp(회귀계수)`\*\*를 계산하면 됩니다.

-----

## 3\. 실전 문제 유형별 해결법 (코드 포함)

가상의 데이터를 만들어서 바로 보여드릴게요.

### 📊 데이터 준비

```python
import pandas as pd
import numpy as np
from statsmodels.formula.api import logit

# 가상 데이터 생성
df = pd.DataFrame({
    'Churn': [1, 0, 1, 0, 1, 0, 0, 1, 1, 0], # 1: 이탈, 0: 유지
    'PlanType': [1, 0, 1, 0, 1, 0, 0, 1, 0, 0], # 1: 고급요금제, 0: 일반
    'Age': [50, 20, 45, 22, 55, 25, 30, 48, 52, 28]
})

# 모델 학습
model = logit('Churn ~ PlanType + Age', data=df).fit()
# print(model.summary())
```

### ❓ 문제 유형 A: "PlanType이 1인 고객이 0인 고객보다 이탈할 오즈비를 구하시오."

이 말은 \*\*"PlanType 변수의 오즈비를 구하라"\*\*는 말과 100% 똑같습니다.
(왜냐하면 회귀계수는 변수가 \*\*'1단위 증가할 때'\*\*의 변화량이기 때문입니다. 0에서 1이 되는 것도 1단위 증가죠.)

**풀이 코드:**

```python
# 1. PlanType의 회귀계수(coef) 추출
coef_plan = model.params['PlanType']

# 2. 지수함수(exp)를 씌워 오즈비 계산
odds_ratio = np.exp(coef_plan)

print(f"PlanType의 오즈비: {round(odds_ratio, 4)}")
```

  * **해석:** 만약 답이 **2.5**라면? -\> "PlanType 1인 사람은 0인 사람보다 이탈할 확률(오즈)이 **2.5배 높다**."

-----

### ❓ 문제 유형 B: "이탈 확률이 0.3 이상인 고객 수를 구하시오."

`statsmodels`의 로지스틱 회귀 모델에서 `predict()`를 쓰면 \*\*자동으로 확률(0\~1 사이 값)\*\*이 나옵니다. (sklearn과 다름\!)

**풀이 코드:**

```python
# 1. 이탈 확률 예측 (0.82, 0.11, ... 이런 식으로 나옴)
pred_probs = model.predict(df) 

# 2. 조건 필터링 (0.3 이상)
target_customers = pred_probs[pred_probs >= 0.3]

# 3. 개수 세기 (len 사용)
count = len(target_customers)

print(f"확률 0.3 이상인 고객 수: {count}명")
```

-----

### ❓ 문제 유형 C: "Age가 10살 증가할 때의 오즈비를 구하시오."

가끔 이렇게 \*\*"1단위가 아니라 n단위 증가할 때"\*\*를 묻기도 합니다.

  * 1살 증가 오즈비 = $\exp(\text{coef})$
  * 10살 증가 오즈비 = $\exp(\text{coef} \times 10)$

**풀이 코드:**

```python
coef_age = model.params['Age']
odds_ratio_10 = np.exp(coef_age * 10) # 계수에 10을 곱하고 exp

print(f"나이 10살 증가 시 오즈비: {round(odds_ratio_10, 4)}")
```

-----

## 📝 시험장용 치트시트 (복사해서 외우세요\!)

```python
import numpy as np
from statsmodels.formula.api import logit

# 1. 모델 생성 및 학습
model = logit('Target ~ var1 + var2', data=df).fit()

# 2. 회귀계수 확인 (coef)
print(model.params['var1'])

# 3. 오즈비(Odds Ratio) 계산 (문제: var1의 오즈비는?)
# 공식: exp(coef)
or_val = np.exp(model.params['var1'])

# 4. 확률 예측 및 개수 세기 (문제: 확률 0.5 이상인 개수는?)
probs = model.predict(df)
count = sum(probs >= 0.5) # 또는 len(probs[probs >= 0.5])
```

이것만 알면 3유형 로지스틱 회귀 문제는 다 풀 수 있습니다\!

`model.params['Group[T.Treatment]']`라는 이름이 조금 복잡하고 낯설어 보일 수 있습니다.

이것은 **`statsmodels` 라이브러리가 범주형(문자열) 데이터를 처리하는 독특한 방식** 때문입니다. 하나씩 쪼개서 아주 쉽게 설명해 드릴게요.

---

### 1. 왜 이런 이름이 생겼나요? (자동 더미 변수화)

우리가 가진 데이터 `df['Group']`에는 **'Control'**과 **'Treatment'**라는 두 가지 문자열이 들어있습니다.

하지만 회귀분석 수식은 **숫자**만 계산할 수 있습니다. 그래서 `statsmodels`는 우리가 시키지 않아도 내부적으로 이렇게 변환합니다.

1.  **문자열 발견!**: "어? `Group` 컬럼에 글자가 있네?"
2.  **기준(Reference) 정하기**: "알파벳 순서로 **'Control'**이 **'Treatment'**보다 먼저네? 그럼 **'Control'을 기준(0)**으로 잡자."
3.  **변수 만들기**: "그럼 남은 **'Treatment'**가 1인 변수를 만들자."

이때 만들어진 변수의 이름이 바로 **`Group[T.Treatment]`**입니다.

* **Group**: 원래 컬럼 이름
* **T.**: Treatment Coding의 약자로, "이 변수는 범주형입니다"라고 표시하는 꼬리표
* **Treatment**: 현재 1로 표시하고 있는 값

---

### 2. 'Control'은 어디 갔나요? (기준 집단)

이게 가장 중요한 핵심입니다. **기준이 되는 집단(Reference Group)은 회귀식에서 사라집니다(숨겨집니다).**



* **Control (대조군):** 기준점입니다. (식에서 생략됨, $X=0$)
* **Treatment (투약군):** 비교 대상입니다. (식에 등장함, $X=1$)

따라서 `Group[T.Treatment]`의 회귀계수(coef)는 **"Control 그룹(0)과 비교했을 때, Treatment 그룹(1)은 얼마나 차이가 나는가?"**를 의미합니다.

---

### 3. 실제 해석 예시

만약 `coef = 1.5`가 나왔다면?

* **잘못된 해석:** "Treatment 그룹의 점수는 1.5점이다."
* **올바른 해석:** "Treatment 그룹은 **Control 그룹보다** (로그 오즈가) **1.5만큼 높다.**"

> **💡 요약 (시험용 암기)**
>
> * `statsmodels`는 문자열 변수를 넣으면 **알파벳 순서가 빠른 놈을 기준(0)**으로 삼는다.
> * 결과표(`summary`)에 나온 놈은 **기준과 비교되는 놈(1)**이다.
> * `Group[T.Treatment]` 계수는 **"기준(Control) 대비 효과"**이다.

결론부터 말씀드리면 **"독립변수가 '범주형(Categorical)'일 때만"** `C()`로 감싸야 합니다.

독립변수가 \*\*수치형(연속형)\*\*이라면 `C()`를 쓰지 말고 변수명만 그대로 써야 합니다. 시험에서 헷갈리지 않게 딱 정리해 드릴게요.

-----

### ⚡️ `C()` 사용 기준표 (무조건 암기\!)

`C()`는 \*\*"Categorical(범주형)"\*\*의 약자입니다.

| 변수 종류 | 데이터 예시 | `C()` 사용 여부 | 코드 작성법 |
| :--- | :--- | :--- | :--- |
| **범주형** (문자) | 남/여, A반/B반, 투약군/대조군 | **필수 (O)** | `C(Gender)`, `C(Group)` |
| **범주형** (숫자) | 1반/2반/3반, 등급(1,2,3) | **필수 (O)** | `C(Class)`, `C(Rank)` |
| **수치형** (숫자) | 나이(25, 30), 키(175), 온도(36.5) | **사용 안 함 (X)** | `Age`, `Height` |

-----

### 🧐 왜 구분해야 하나요? (중요)

**1. 숫자로 된 범주형 (예: 1반, 2반, 3반)**

  * **`C(Class)`라고 쓰면:** "아, 이건 숫자가 아니라 \*\*'이름표'\*\*구나\!"라고 인식해서, 1반 vs 2반 vs 3반을 각각 비교합니다. (올바름)
  * **그냥 `Class`라고 쓰면:** "아, 이건 **숫자**구나\! 3반은 1반보다 **3배 더 강력**하구나\!"라고 착각하고 계산합니다. (틀림)

**2. 진짜 수치형 (예: 나이 20세, 30세)**

  * **그냥 `Age`라고 쓰면:** "나이가 많아질수록 결과가 어떻게 변하는지(기울기)"를 봅니다. (올바름)
  * **`C(Age)`라고 쓰면:** 20살 그룹, 21살 그룹... 80살 그룹까지 **모든 나이를 별개의 그룹**으로 쪼개서 분석합니다. (모델이 엉망이 됨)

-----

### 📝 시험 문제 유형별 적용

#### 1\. 분산분석 (ANOVA) ➡️ 99% `C()` 사용

ANOVA는 애초에 \*\*"그룹 간 차이"\*\*를 보는 것이므로, 독립변수가 항상 **범주형**입니다.

```python
# Diet(A, B, C)에 따른 체중 변화
model = ols('Weight_Loss ~ C(Diet)', data=df).fit()
```

#### 2\. 회귀분석 (Regression) ➡️ 섞여 있음

독립변수 성격에 따라 다릅니다.

```python
# 나이(수치), BMI(수치), 성별(범주)로 혈압 예측
# Age, BMI는 그냥 쓰고, Sex만 C()를 씌움
model = ols('BP ~ Age + BMI + C(Sex)', data=df).fit()
```

> **💡 꿀팁:**
> `statsmodels`는 데이터가 \*\*'문자열(String)'\*\*이면 `C()`를 안 써도 알아서 범주형으로 인식해 주긴 합니다.
> 하지만, \*\*데이터가 숫자(1, 2, 3)로 되어 있는 범주형(등급, 반 번호)\*\*일 때는 **반드시 `C()`를 써야 합니다.**
>
> **결론:** 헷갈리면 \*\*"그룹을 나누는 변수다 싶으면 무조건 `C()`를 씌운다"\*\*고 생각하세요.

`anova_lm`의 결과로 나오는 \*\*분산분석표(ANOVA Table)\*\*는 처음 보면 숫자가 많아서 복잡해 보이지만, 시험에서 물어보는 건 정해져 있습니다.

**"어디를 봐야 정답을 찾을 수 있는지"** 딱 집어 드릴게요.

-----

### 📊 ANOVA 테이블 해부도

`statsmodels`로 출력한 표는 보통 아래와 같이 생겼습니다.

| Index (행 이름) | **df** (자유도) | **sum\_sq** (제곱합) | **mean\_sq** (평균제곱) | **F** (F통계량) | **PR(\>F)** (p-value) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C(Diet)** <br>(요인/그룹) | 2.0 | 355.5 | 177.75 | **11.5** | **0.0003** |
| **Residual** <br>(오차/잔차) | 147.0 | 2205.0 | 15.0 | NaN | NaN |

-----

### 1\. 핵심 컬럼 3대장 (시험에 무조건 나옴)

시험 문제에서 "무엇을 구하시오"라고 할 때 봐야 할 곳입니다.

#### ① `F` (F-통계량)

  * **위치:** 그룹 행(`C(Diet)`)의 **F** 컬럼
  * **의미:** (그룹 간 차이) ÷ (그룹 내 차이)
  * **해석:** 이 숫자가 **클수록** "그룹 간의 차이가 확실하다"는 뜻입니다.
  * **시험 문제:** "F-통계량을 구하시오" ➡️ **`table.loc['C(Diet)', 'F']`**

#### ② `PR(>F)` (p-value, 유의확률) ⭐가장 중요

  * **위치:** 그룹 행(`C(Diet)`)의 **PR(\>F)** 컬럼
  * **이름의 뜻:** "F-통계량이 이 값보다 클(\>) 확률(Probability)" ➡️ 즉, **p-value**입니다.
  * **해석:**
      * **0.05보다 작으면:** "그룹 간 평균 차이가 **있다**." (귀무가설 기각)
      * **0.05보다 크면:** "그룹 간 평균 차이가 **없다**." (귀무가설 채택)
  * **시험 문제:** "p-값을 구하시오" ➡️ **`table.loc['C(Diet)', 'PR(>F)']`**

#### ③ `sum_sq` (제곱합, Sum of Squares)

  * **위치:**
      * **SSR (처리 제곱합):** `C(Diet)` 행의 `sum_sq` (그룹 간 차이의 총량)
      * **SSE (잔차 제곱합):** `Residual` 행의 `sum_sq` (설명 안 되는 오차의 총량)
  * **시험 문제:** "잔차 제곱합(SSE)을 구하시오" ➡️ **`table.loc['Residual', 'sum_sq']`**

-----

### 2\. 나머지 컬럼 (개념 이해용)

#### ④ `df` (자유도, Degrees of Freedom)

  * **C(Diet)의 df:** (그룹 수 - 1). 예: 그룹이 3개(A, B, C)면 2.
  * **Residual의 df:** (전체 데이터 수 - 그룹 수).

#### ⑤ `mean_sq` (평균제곱, Mean Squares)

  * **계산법:** `sum_sq` ÷ `df`
  * 분산(Variance)을 추정한 값입니다.
  * F-통계량은 결국 `mean_sq(그룹)` ÷ `mean_sq(잔차)`로 계산됩니다.

-----

### ⚡️ 한 장 요약 (시험장용)

코드로 값을 뽑아낼 때 \*\*인덱스(행 이름)\*\*를 틀리지 않도록 주의하세요.

```python
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

model = ols('Weight_Loss ~ C(Diet)', data=df).fit()
table = anova_lm(model)

# 1. F-통계량
print(table.loc['C(Diet)', 'F'])

# 2. p-value (PR(>F))
print(table.loc['C(Diet)', 'PR(>F)'])

# 3. 잔차 제곱합 (SSE) -> 행 이름이 'Residual'임에 주의!
print(table.loc['Residual', 'sum_sq'])
```

**팁:** `print(table)`을 먼저 해서 표의 행/열 이름을 눈으로 확인하고, `loc`을 사용하면 실수를 줄일 수 있습니다.

시험 준비를 위해 `statsmodels` 라이브러리를 활용한 선형회귀(OLS)와 로지스틱 회귀(Logit) 핵심 코드를 먼저 정리해 드리겠습니다. 그 후, 해당 개념을 확인하는 연습 문제를 풀어보세요.

### 📚 시험 대비: statsmodels 핵심 코드 정리

빅데이터 분석기사 실기 등 파이썬 기반 통계 분석 시험에서는 `statsmodels.formula.api`를 사용하는 것이 수식을 직관적으로 작성할 수 있어 유리합니다.

#### 1\. 라이브러리 임포트

```python
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols, logit
```

#### 2\. 다중 선형 회귀 (OLS: Ordinary Least Squares)

연속형 종속변수($Y$)를 예측할 때 사용합니다.

```python
# 모델 정의 및 학습 (Formula: '종속변수 ~ 독립변수1 + 독립변수2')
# data: 데이터프레임 이름
model = ols('target ~ feature1 + feature2', data=df).fit()

# 결과 요약 (회귀계수, p-value, R-squared 확인)
print(model.summary())

# 회귀계수만 확인
print(model.params)

# 예측 (테스트 데이터)
pred = model.predict(test_df)
```

✅ 제3유형 (통계): 무조건 ols 쓰세요! (sm.OLS X)
시험 문제에서 "회귀계수의 p-value를 구하시오" 또는 **"유의하지 않은 변수를 찾으시오"**라고 묻습니다.

#### 3\. 로지스틱 회귀 (Logit)

이진 분류(0 또는 1) 종속변수($Y$)를 예측할 때 사용합니다.

```python
# 모델 정의 및 학습
model = logit('target ~ feature1 + feature2', data=df).fit()

# 결과 요약 (Pseudo R-squared, Log-Likelihood 확인)
print(model.summary())

# ★ 중요: 오즈비(Odds Ratio) 구하기
# model.params는 '로그 오즈(Log-Odds)' 값이므로지수함수(exp)를 취해야 오즈비가 됨
odds_ratios = np.exp(model.params)
print(odds_ratios)

# 예측 (결과는 0~1 사이의 확률값으로 반환됨)
pred_prob = model.predict(test_df)

# 확률을 0/1 클래스로 변환 (임계값 0.5 기준)
pred_class = np.where(pred_prob > 0.5, 1, 0)
```

-----

그럼 이제 학습한 내용을 바탕으로 실전 감각을 익혀볼까요?

선형회귀와 로지스틱 회귀의 개념, 코드 사용법, 결과 해석에 대한 퀴즈입니다.

빅데이터 분석기사 실기 시험에서 **모델 학습 후(`.fit()`)** 뽑아낼 수 있는 핵심 속성(Attribute)들을 라이브러리별로 정리해 드립니다.

시험장에서는 **`dir(model)`** 명령어를 치면 모든 속성을 볼 수 있지만, 시간이 없으니 아래 핵심 속성들은 외워가는 게 좋습니다.

-----

### 1\. 📉 Statsmodels (제3유형 - 통계용)

주로 \*\*`ols` (선형회귀)\*\*와 \*\*`logit` (로지스틱회귀)\*\*를 사용한 뒤, **`.fit()`으로 생성된 객체**에서 뽑아냅니다.

```python
from statsmodels.formula.api import ols, logit
# 학습 완료된 객체
model = ols('y ~ x1 + x2', data=df).fit() 
```

| 속성 이름 | 설명 | 사용 예시 (코드) | 대상 모델 |
| :--- | :--- | :--- | :--- |
| **`params`** | **회귀계수** (Coefficient) | `model.params['x1']` | 전체 |
| **`pvalues`** | **p-값** (유의확률) | `model.pvalues['x1']` | 전체 |
| **`rsquared`** | **결정계수** ($R^2$) | `model.rsquared` | OLS (선형회귀) |
| **`rsquared_adj`**| **수정된 결정계수** | `model.rsquared_adj` | OLS (선형회귀) |
| **`prsquared`** | **유사 결정계수** (Pseudo $R^2$) | `model.prsquared` | Logit (로지스틱) |
| **`conf_int()`** | **신뢰구간** | `model.conf_int()` | 전체 |
| **`resid`** | **잔차** (실제값 - 예측값) | `model.resid` | 전체 |
| **`fittedvalues`**| **예측값** (학습 데이터에 대한) | `model.fittedvalues` | 전체 |
| **`aic` / `bic`** | 모델 적합도 지수 | `model.aic`, `model.bic` | 전체 |

> **💡 꿀팁:** `model.summary()`를 찍어보면 이 모든 게 표로 나옵니다. 하지만 문제에서 \*\*"값 하나만 제출하시오"\*\*라고 하면 위 속성을 써서 숫자로 딱 뽑아야 합니다.

-----

### 2\. 🤖 Scikit-Learn (제2유형 - 머신러닝용)

Scikit-learn 모델은 학습 후 속성 이름 뒤에 \*\*언더바(`_`)\*\*가 붙는 것이 특징입니다.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

model = RandomForestClassifier().fit(X, y)
lr = LinearRegression().fit(X, y)
```

| 속성 이름 | 설명 | 사용 예시 (코드) | 대상 모델 |
| :--- | :--- | :--- | :--- |
| **`feature_importances_`** | **변수 중요도** (어떤 컬럼이 중요한지) | `model.feature_importances_` | 트리 계열 (RandomForest, XGBoost) |
| **`coef_`** | **회귀계수** (기울기) | `lr.coef_` | 선형회귀, 로지스틱회귀 |
| **`intercept_`** | **절편** (상수항, y절편) | `lr.intercept_` | 선형회귀, 로지스틱회귀 |
| **`classes_`** | **클래스 종류** (0, 1 등) | `model.classes_` | 분류 모델 (Classifier) |
| **`n_features_in_`** | 학습에 사용된 **변수(컬럼) 개수** | `model.n_features_in_` | 전체 |
| **`feature_names_in_`** | 학습에 사용된 **변수(컬럼) 이름** | `model.feature_names_in_` | 전체 (최신 버전) |

-----

### ⚡️ 시험장 긴급 상황 대처법 (`dir` 함수)

시험치다가 속성 이름이 기억 안 나면? **`dir()`** 함수를 쓰세요\!
해당 객체가 가진 모든 기능과 속성을 리스트로 보여줍니다.

```python
# 예: result = ols(...).fit() 일 때
print(dir(result))

# 출력 예시:
# [..., 'aic', 'bic', 'conf_int', 'f_test', 'fittedvalues', 
#  'params', 'pvalues', 'predict', 'rsquared', 'summary', ...]
```

**핵심 요약:**

1.  **3유형(통계):** `params`(계수), `pvalues`(p값), `rsquared`(R2) 무조건 암기.
2.  **2유형(ML):** `feature_importances_`(중요도) 정도만 알면 됨. (나머지는 `predict`만 잘하면 됨)

\*\*"유의하지 않다"\*\*는 통계적으로 \*\*"p-value가 0.05보다 크다(\> 0.05)"\*\*는 뜻입니다.
(귀무가설을 기각할 수 없음 = 우연일 수 있음 = 영향력이 확실하지 않음)

`statsmodels`에서 이를 찾아내는 방법은 **`model.pvalues`** 속성을 이용하는 것입니다. 시험에 자주 나오는 패턴 2가지로 정리해 드릴게요.

-----

### 1\. 유의하지 않은 변수의 "개수" 구하기

시험 문제: *"유의수준 0.05 하에서 유의하지 않은 설명변수의 개수를 구하시오. (단, 절편항은 제외)"*

```python
# 1. 모델 학습
model = ols('y ~ x1 + x2 + x3', data=df).fit()

# 2. 모든 변수의 p-value 확인
# print(model.pvalues)

# 3. [핵심] p-value가 0.05보다 큰 것만 필터링 (True=1, False=0)
# Intercept(절편)는 보통 설명변수 개수에서 빼므로 [1:]로 슬라이싱합니다.
target_vars = model.pvalues[1:] # 0번 인덱스는 Intercept
cnt = sum(target_vars > 0.05)

print(cnt)
```

### 2\. 유의하지 않은 변수의 "이름" 찾기

시험 문제: *"가장 유의하지 않은(p-값이 가장 큰) 변수의 이름을 적으시오."*

```python
# 1. 설명변수(절편 제외)의 p-value만 추출
vars_p = model.pvalues[1:]

# 2. p-값이 0.05보다 큰 변수들의 이름 출력
not_signif = vars_p[vars_p > 0.05].index
print("유의하지 않은 변수들:", list(not_signif))

# 3. 가장 p-값이 큰(가장 쓸모없는) 변수 하나 찾기
worst_var = vars_p.idxmax()
print("가장 유의하지 않은 변수:", worst_var)
```

-----

### ⚡️ 요약 (시험장용)

1.  \*\*`model.pvalues`\*\*를 쓴다.
2.  **`> 0.05`** 조건이 "유의하지 않은" 것이다.
3.  \*\*`model.pvalues[1:]`\*\*를 해서 \*\*Intercept(절편)\*\*를 빼는 것을 잊지 않는다. (문제에서 "절편 포함"이라 하면 그냥 씀)