빅데이터 분석기사 실기 시험에서 \*\*`idxmax()`\*\*는 \*\*"가장 큰 값을 가진 녀석의 이름(인덱스)이 뭐야?"\*\*라고 물을 때 쓰는 최고의 치트키입니다.

시험장에서 헷갈리지 않게 딱 정리해 드립니다.

-----

# 🏆 `idxmax()` 완벽 정리

### 1\. 한 줄 요약

  * **`max()`**: **"1등 점수가 몇 점이야?"** (값 반환)
  * **`idxmax()`**: **"1등이 누구야?"** (인덱스 반환)

### 2\. 가장 많이 쓰는 패턴 (Series)

주로 `groupby`를 한 뒤에 많이 씁니다.

```python
import pandas as pd

# 데이터: 학생별 점수
s = pd.Series([80, 90, 50], index=['철수', '영희', '민수'])

# 1. max() -> 90 (점수)
print(s.max())

# 2. idxmax() -> '영희' (이름)
print(s.idxmax()) 
```

-----

### 3\. 시험용 필승 패턴 (데이터프레임에서 행 가져오기)

"가장 매출이 높은 도시의 모든 정보를 출력하시오" 같은 문제에서 씁니다.

**방법 A: 정렬해서 뽑기 (익숙함)**

```python
# 내림차순 정렬 -> 맨 위(0번째) 뽑기
top_city = df.groupby('City')['Sales'].sum().sort_values(ascending=False).index[0]
```

**방법 B: `idxmax` 쓰기 (빠름 & 간지남)**

```python
# 합계가 가장 큰 '인덱스(도시명)'를 바로 가져옴
top_city = df.groupby('City')['Sales'].sum().idxmax()
```

-----

### 4\. 주의할 점 (`loc`과 찰떡궁합)

`idxmax()`로 인덱스를 구했다면, 그 인덱스로 **전체 데이터**를 가져올 때 `loc`을 씁니다.

```python
# 1. 가장 비싼 제품의 '인덱스 번호'를 찾음
expensive_idx = df['Price'].idxmax()

# 2. 그 인덱스에 해당하는 '행 전체'를 가져옴
print(df.loc[expensive_idx])
```

### ⚡️ 요약 (Cheat Sheet)

| 함수 | 의미 | 결과 예시 |
| :--- | :--- | :--- |
| **`.max()`** | 최댓값 **(Value)** | `100` |
| **`.idxmax()`**| 최댓값의 **위치(Index)** | `'City_A'` 또는 `3` (행번호) |
| **`.min()`** | 최솟값 **(Value)** | `0` |
| **`.idxmin()`**| 최솟값의 **위치(Index)** | `'City_Z'` 또는 `10` (행번호) |

**"값이 아니라 이름(누구?)을 찾을 땐 `idxmax`\!"** 이것만 기억하세요.

\*\*K-S 검정(Kolmogorov-Smirnov Test)\*\*은 \*\*"두 데이터의 분포가 같은지, 다른지"\*\*를 비교하는 비모수 검정 방법입니다.

빅데이터분석기사 실기(3유형)에서는 주로 **① 정규성 검정(데이터가 많을 때)** 또는 **② 두 집단의 분포 비교** 용도로 사용됩니다.

-----

### 1\. 핵심 개념 (그림으로 이해하기)

K-S 검정은 \*\*두 누적 분포 곡선(CDF) 사이의 거리가 가장 멀리 떨어진 지점($D$ 통계량)\*\*을 찾습니다.

  * **거리가 멀다 ($D$가 크다):** 두 분포가 **다르다**.
  * **거리가 가깝다 ($D$가 작다):** 두 분포가 **비슷하다(같다)**.

-----

### 2\. 종류 및 코드 (시험용)

#### ① 1표본 K-S 검정 (`stats.kstest`)

  * **목적:** "내 데이터가 \*\*특정 이론적 분포(예: 정규분포)\*\*를 따르는가?" (주로 정규성 검정용)
  * **주의:** `kstest`의 `'norm'`은 \*\*표준정규분포(평균=0, 분산=1)\*\*와 비교합니다. 따라서 데이터를 넣을 때 \*\*반드시 표준화(Scaling)\*\*하거나, 평균/표준편차를 알려줘야 합니다.

<!-- end list -->

```python
from scipy import stats
import numpy as np

# 데이터 생성 (평균 50, 표준편차 10인 정규분포)
data = np.random.normal(50, 10, 1000)

# 방법 1: 데이터를 표준화한 뒤 검정 (추천)
# (데이터 - 평균) / 표준편차
data_scaled = (data - np.mean(data)) / np.std(data, ddof=1)
stat, p_val = stats.kstest(data_scaled, 'norm')

print(f"p-value: {p_val:.4f}")
# p > 0.05 이면 "정규분포를 따른다"
```

#### ② 2표본 K-S 검정 (`stats.ks_2samp`)

  * **목적:** "두 집단 A와 B가 **같은 분포**에서 나왔는가?" (분산, 평균, 모양 등 포괄적 비교)
  * **특징:** t-검정 전에 두 집단의 분포 모양이 비슷한지 볼 때 쓸 수 있습니다.

<!-- end list -->

```python
from scipy import stats

group_A = [1, 2, 3, 4, 5]
group_B = [1, 2, 3, 4, 100] # 100 때문에 분포가 다름

# 두 집단 비교
stat, p_val = stats.ks_2samp(group_A, group_B)

print(f"p-value: {p_val:.4f}")
```

-----

### 3\. 가설 및 해석 (공통)

  * **귀무가설 ($H_0$):** 두 분포는 **같다** (또는 데이터가 정규분포를 따른다).
  * **대립가설 ($H_1$):** 두 분포는 **다르다** (정규분포를 따르지 않는다).

> **`p-value > 0.05`** ➡️ **"분포가 같다" (정규성 만족 / 두 집단 동일)**
> **`p-value <= 0.05`** ➡️ **"분포가 다르다"**

-----

### ⚡️ 시험장 꿀팁 (Shapiro vs K-S)

시험에서 "정규성 검정을 수행하시오"라고 했을 때 뭘 써야 할까요?

1.  **`stats.shapiro` (샤피로-윌크):**
      * **데이터가 적을 때 (약 5000개 미만)** 가장 정확합니다.
      * 시험에서는 대부분 이거 쓰면 됩니다.
2.  **`stats.kstest` (K-S 검정):**
      * **데이터가 매우 많을 때 (2000\~5000개 이상)** 사용합니다.
      * 샤피로는 데이터가 너무 많으면 무조건 "정규분포 아님(p\<0.05)"이라고 억지를 부리는 경향이 있어서, 그때 K-S를 씁니다.

**결론:** 문제에서 꼭 집어서 "K-S 검정을 쓰시오"라고 하지 않는 이상, 정규성 검정은 \*\*`shapiro`\*\*가 1순위입니다.

빅데이터 분석기사 실기에서 데이터 추출의 핵심인 4가지 개념(`index`, `loc`, `iloc`, `idxmax`)을 \*\*"행(가로) vs 열(세로) 중 어디를 뽑느냐"\*\*를 중심으로 정리해 드립니다.

시험장에서는 이 규칙 하나만 기억하세요:

> **모든 선택 명령어는 `[행, 열]` 순서입니다.** (행이 먼저, 열이 나중)

-----

## 데이터 추출 최종 정리

### ⚡️ 한 눈에 보는 비교표

| 구분 | 이름 | 기준 | 문법 (행, 열) | 비유 |
| :--- | :--- | :--- | :--- | :--- |
| **`index`** | 인덱스 | **행의 이름표** | `df.index` | 아파트 호수 (101호, 102호...) |
| **`loc`** | 라벨 선택 | **이름(Name)** | `df.loc['행이름', '열이름']` | "철수네 집(행)의 안방(열)을 찾아라" |
| **`iloc`** | 위치 선택 | **번호(Number)** | `df.iloc[행번호, 열번호]` | "위에서 3번째 집(행)의 2번째 방(열)" |
| **`idxmax`**| 1등 찾기 | **최댓값의 인덱스** | `series.idxmax()` | "1등 한 학생의 **이름**은?" |

-----

### 1\. `index` (행의 이름표)

데이터프레임의 **가로줄(행) 하나하나에 붙어있는 이름표**입니다.

  * **기본값:** 따로 설정 안 하면 `0, 1, 2, 3...` (자동 번호)
  * **설정값:** `Date`(날짜)나 `ID` 등을 인덱스로 지정할 수 있음.

<!-- end list -->

```python
print(df.index)
# 결과: Index(['User_A', 'User_B', 'User_C'], dtype='object')
```

-----

### 2\. `loc` : "이름"으로 뽑기 (가장 중요 ⭐)

**`loc[행이름, 열이름]`**

가장 직관적입니다. 눈에 보이는 **글자(라벨)** 그대로 찾습니다.

  * **행 뽑기:** `df.loc['User_A']` (User\_A 행 전체)
  * **열 뽑기:** `df.loc[:, 'Age']` (모든 행의 Age 열)
      * *(주의: 행 자리에 전체를 뜻하는 콜론(`:`)을 찍어야 함)*
  * **행+열 뽑기:** `df.loc['User_A', 'Age']` (User\_A의 나이)
  * **범위:** `df.loc['A':'C']` (**끝인 C도 포함됨\!**)

-----

### 3\. `iloc` : "번호(순서)"로 뽑기

**`iloc[행번호, 열번호]`**

이름 상관없이 무조건 \*\*컴퓨터 기준 순서(0부터 시작)\*\*로 찾습니다.

  * **행 뽑기:** `df.iloc[0]` (맨 첫 번째 줄)
  * **열 뽑기:** `df.iloc[:, -1]` (모든 행의 **맨 마지막 열**) -\> **Target 분리할 때 필수\!**
  * **범위:** `df.iloc[0:3]` (**끝인 3은 제외됨\!** 0, 1, 2만 나옴)

-----

### 4\. `idxmax` : "최댓값이 있는 곳의 행 이름(Index)"

**"값(Value)"이 아니라 "위치(Index)"를 반환합니다.**

  * **상황:** "가장 매출이 높은 **도시**는 어디인가?"
  * **사용법:**
    ```python
    # 1. 도시별 매출 합계 계산
    s = df.groupby('City')['Sales'].sum()

    # s 출력 결과:
    # Seoul    500
    # Busan    300
    # Daegu    600  <- 여기가 최대 (값: 600, 인덱스: Daegu)

    print(s.max())    # 결과: 600 (점수)
    print(s.idxmax()) # 결과: 'Daegu' (이름) -> 정답!
    ```

-----

### 📝 시험장용 암기 노트

1.  **조건으로 행을 뽑고 싶다?** 👉 **`loc`**
      * `df.loc[df['Age'] >= 20]`
2.  **마지막 열(Target)만 뚝 떼고 싶다?** 👉 **`iloc`**
      * `y = df.iloc[:, -1]`
3.  **1등인 녀석의 이름을 알고 싶다?** 👉 **`idxmax`**
      * `best_city = df.groupby(...).sum().idxmax()`

이것만 기억하면 데이터 추출 문제는 완벽합니다\!

빅데이터 분석기사 실기 시험에서 가장 헷갈리기 쉬운 **카이제곱 검정 2종류**와 **분류 모델 평가지표**를 시험장용으로 딱 정리해 드립니다.

-----

# 1\. 🔲 카이제곱 검정 (제3유형 - 통계)

두 함수는 이름이 비슷하지만 **용도가 완전히 다릅니다.**

### ① `chi2_contingency` (독립성 검정)

  * **목적:** "두 범주형 변수(A, B)가 서로 **관련이 있는지(독립인지)**" 확인할 때.
  * **입력:** **교차표 (Crosstab)** (2x2, 3x2 등 표 형태)
  * **예시:** "성별(남/여)에 따라 생존 여부(생존/사망)가 다른가?"

<!-- end list -->

```python
from scipy import stats
import pandas as pd

# 1. 교차표 만들기 (필수!)
ct = pd.crosstab(df['성별'], df['생존여부'])

# 2. 검정 수행 (반환값 4개 순서 암기: 통계량, p값, 자유도, 기대빈도)
chi2, p_val, dof, expected = stats.chi2_contingency(ct)

print(round(p_val, 4))
```

### ② `chisquare` (적합도 검정)

  * **목적:** "한 변수의 데이터 분포가 \*\*특정 비율(가설)\*\*과 일치하는지" 확인할 때.
  * **입력:** **관측 빈도 리스트**(`f_obs`)와 **기대 빈도 리스트**(`f_exp`)
  * **예시:** "주사위를 던졌는데 1\~6이 모두 1/6 확률로 나왔는가?", "성비가 남:여 = 6:4가 맞는가?"

<!-- end list -->

```python
from scipy import stats

# 1. 관측 빈도 (실제 데이터 개수)
f_obs = [60, 40] # 남자 60명, 여자 40명

# 2. 기대 빈도 (가설에 따른 개수 - 예: 50:50이라면)
f_exp = [50, 50]

# 3. 검정 수행 (반환값 2개)
stat, p_val = stats.chisquare(f_obs, f_exp)

print(round(p_val, 4))
```

-----

# 2\. 🎯 모델 평가 (제2, 3유형 - 머신러닝)

### ③ `accuracy_score` (정확도)

  * **목적:** "분류 모델이 **얼마나 정답을 잘 맞췄는지**" (전체 중 맞은 비율)
  * **입력:** \*\*실제 정답값(`y_true`)\*\*과 **예측된 클래스값(`y_pred`)**
      * ⚠️ **주의:** 확률(`predict_proba`)이 아니라 \*\*0, 1 같은 클래스(`predict`)\*\*를 넣어야 합니다\!

<!-- end list -->

```python
from sklearn.metrics import accuracy_score
import numpy as np

# 1. 예측값 준비 (predict 사용)
# 만약 statsmodels라면 predict 결과(확률)를 0/1로 변환해야 함
pred_class = [1, 0, 1, 1, 0] 
y_true = [1, 0, 0, 1, 0]

# 2. 정확도 계산
acc = accuracy_score(y_true, pred_class)

print(round(acc, 2))
```

-----

### ⚡️ 시험장 요약표 (Cheat Sheet)

| 함수 이름 | 라이브러리 | 용도 | 입력값 형태 | 반환값 (핵심) |
| :--- | :--- | :--- | :--- | :--- |
| **`chi2_contingency`** | `scipy.stats` | **독립성** (변수 2개 관계) | **`pd.crosstab` (표)** | `stat`, **`p_val`**, ... |
| **`chisquare`** | `scipy.stats` | **적합도** (비율 맞는지) | **`[관측]`, `[기대]` (리스트)** | `stat`, **`p_val`** |
| **`accuracy_score`** | `sklearn.metrics` | **정확도** (채점) | **`정답`, `예측값(0/1)`** | **`점수(0~1)`** |

이 표만 기억하면 헷갈리지 않고 바로 코드를 작성하실 수 있습니다\!

```python
# chisquare 시험용 예시
from scipy.stats import chisquare
#print(train['Seat_Class'].unique())
count_eco = len(train[train['Seat_Class']=='Economy'])
count_bis = len(train[train['Seat_Class']=='Business'])

f_obs = [count_eco, count_bis]
total = len(train)
f_exp = [total*0.7, total*0.3]

stat, p_val = chisquare(f_obs=f_obs, f_exp=f_exp)
print(round(stat, 3))

# accuracy 시험용 예시
from sklearn.metrics import accuracy_score
pred_proba = model.predict(test)
pred_class = np.where(pred_proba > 0.5, 1, 0)
accuracy = accuracy_score(test['Satisfied'], pred_class)
print(round(accuracy, 2))
```

빅데이터 분석기사 실기 3유형에서 \*\*합동 분산 추정량 (Pooled Variance Estimator)\*\*은 \*\*"두 집단의 분산이 같다고 가정할 때(등분산), 두 분산을 하나로 합쳐서 더 정확하게 추정한 값"\*\*입니다.

시험용으로 핵심만 딱 정리해 드립니다.

-----

### 1\. 🎯 언제 사용하나요?

  * **상황:** **독립표본 t-검정**을 할 때.
  * **조건:** \*\*"두 집단의 분산이 같다(등분산)"\*\*는 가정을 만족할 때. (`equal_var=True`)
  * **목적:** 두 개의 분산($s_A^2, s_B^2$)을 따로 쓰지 않고, 데이터 개수($n$)를 반영해 **가중 평균**을 내서 하나의 공통 분산($s_p^2$)을 만듭니다.

-----

### 2\. 🧮 공식 (시험용 암기)

파이썬 라이브러리에 이 값을 한 번에 구해주는 함수가 없기 때문에, **공식을 외워서 코드로 짜야 합니다.**

$$s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{(n_1 - 1) + (n_2 - 1)}$$

  * $n_1, n_2$: 각 그룹의 데이터 개수
  * $s_1^2, s_2^2$: 각 그룹의 **표본 분산** (`ddof=1`)
  * 분모: 전체 자유도 ($n_1 + n_2 - 2$)

-----

### 3\. 💻 실전 코드 (복사해서 외우세요)

문제: *"두 집단 A와 B의 합동 분산 추정량을 구하시오."*

```python
import numpy as np

# 데이터 예시
group_A = [10, 12, 11, 14, 15]
group_B = [20, 22, 19, 21, 23, 25]

# 1. 각 집단의 표본 분산 구하기 (★ ddof=1 필수!)
var_A = np.var(group_A, ddof=1)
var_B = np.var(group_B, ddof=1)

# 2. 데이터 개수 구하기
n_A = len(group_A)
n_B = len(group_B)

# 3. 합동 분산 공식 적용
numerator = (n_A - 1) * var_A + (n_B - 1) * var_B  # 분자
denominator = (n_A - 1) + (n_B - 1)                # 분모 (자유도 합)

pooled_var = numerator / denominator

print(f"합동 분산: {pooled_var:.3f}")
```

-----

### ⚡️ 시험장 요약 노트

1.  **언제 쓴다?** 👉 **t-검정에서 등분산 가정(`equal_var=True`)일 때.**
2.  **주의할 점?** 👉 분산 구할 때 **`np.var(..., ddof=1)`** 안 하면 틀림\! (표본 분산이어야 함)
3.  **t-검정 함수와의 관계:**
      * `stats.ttest_ind(A, B, equal_var=True)`를 실행하면 내부적으로 이 **합동 분산**을 써서 계산합니다.
      * 문제에서 \*\*"합동 분산 값을 직접 구하시오"\*\*라고 할 때만 위 공식을 써서 계산하면 됩니다.


** 빅데이터 분석기사 실기 1유형에서 \*\*`pivot_table`\*\*은 **"데이터를 요약해서 표 형태로 보고 싶을 때"** 사용하는 가장 강력한 도구입니다.

엑셀의 **피벗 테이블**과 기능이 100% 똑같습니다.

-----

### 1\. 🖼️ 구조 이해하기 (4요소)

```python
df.pivot_table(index='행_기준', columns='열_기준', values='계산할_값', aggfunc='계산_방법')
```

1.  **`index` (행):** 세로줄(왼쪽)에 올 기준 (예: 연도, 지점)
2.  **`columns` (열):** 가로줄(위쪽)에 올 기준 (예: 월, 상품명)
3.  **`values` (값):** 표 안에 채워질 숫자 데이터 (예: 매출액)
4.  **`aggfunc` (함수):** 값을 어떻게 요약할지 (예: `sum`, `mean`, `count`)
      * **주의:** 기본값(default)은 \*\*`mean` (평균)\*\*입니다. 합계를 원하면 꼭 `'sum'`을 써야 합니다\!

-----

### 2\. 💻 실전 코드 예시

**상황:** "지점(`Store`)별, 상품(`Product`)별로 매출(`Sales`)의 **합계**를 구하시오."

```python
import pandas as pd

# 1. 피벗 테이블 생성
pivot = df.pivot_table(index='Store', 
                       columns='Product', 
                       values='Sales', 
                       aggfunc='sum')

# 결과 예시:
# Product   A     B     C
# Store                  
# Gangnam  100   200   150
# Hongdae  50    NaN   300
```

-----

### 3\. ⚡️ 시험장 꿀팁 & 주의사항

#### ① 결측치 채우기 (`fill_value`)

피벗 테이블을 만들면 데이터가 없는 곳에 `NaN`이 생기기 쉽습니다. (예: 홍대점에서 B상품 안 팔림)
이때 **`fill_value=0`** 옵션을 쓰면 깔끔하게 0으로 채워집니다.

```python
# NaN을 0으로 채우기
df.pivot_table(..., fill_value=0)
```

#### ② `groupby` vs `pivot_table` (언제 뭘 써?)

  * **`groupby`**: 결과가 **세로로 길게** 나옵니다. (Series 형태)
      * 사용: 단순히 집계값만 필요할 때, 나중에 `idxmax` 등을 쓸 때.
  * **`pivot_table`**: 결과가 \*\*행/열이 있는 표(Grid)\*\*로 나옵니다. (DataFrame 형태)
      * 사용: "A 대비 B의 비율" 처럼 행과 열의 관계를 보거나, 시각화할 때.

#### ③ 인덱스 초기화 (`reset_index`)

피벗 테이블을 만들면 `index`로 지정한 컬럼이 실제 인덱스가 되어버립니다.
다시 일반 컬럼으로 쓰고 싶다면 \*\*`.reset_index()`\*\*를 꼭 해주세요.

-----

### 📝 시험장 암기 노트

> **"표로 요약해라"** = **`pivot_table`**
>
> 1.  **행(index)**: 누가?
> 2.  **열(columns)**: 무엇을?
> 3.  **값(values)**: 어떤 수치를?
> 4.  **함수(aggfunc)**: 합(`sum`)? 평균(`mean`)?

이 4가지만 기억하면 됩니다\!

빅데이터 분석기사 실기 시험에서 **결측치(NaN) 처리**는 1유형(전처리)과 2유형(모델링) 모두에서 필수적으로 나오는 항목입니다.

시험장에서 당황하지 않도록 **가장 많이 쓰이는 방법** 위주로 딱 정리해 드립니다.

-----

# 1\. 🔍 결측치 확인하기

가장 먼저 어디에 몇 개가 비어있는지 확인해야 합니다.

```python
# 컬럼별 결측치 개수 확인 (필수!)
print(df.isnull().sum())
```

-----

# 2\. 🗑️ 결측치 제거 (삭제)

데이터가 충분히 많거나, 결측치가 있는 행이 분석에 방해가 될 때 사용합니다.

### ① 행(Row) 삭제 (가장 기본)

결측치가 하나라도 있는 행을 지웁니다.

```python
# 결측치가 있는 행 전체 삭제
df = df.dropna()

# 특정 컬럼(예: 'Age')에 결측치가 있을 때만 해당 행 삭제
df = df.dropna(subset=['Age'])
```

### ② 열(Column) 삭제

특정 컬럼에 결측치가 너무 많아서 아예 그 변수를 안 쓰기로 했을 때.

```python
# 'Cabin' 컬럼 삭제 (axis=1은 열)
df = df.drop(['Cabin'], axis=1)
```

-----

# 3\. 🛠️ 결측치 채우기 (대치)

데이터를 살리면서 빈칸을 채우는 방법입니다. **2유형(모델링)에서 가장 중요합니다.**

### ① 단순 값으로 채우기 (0 또는 특정 값)

```python
# 모든 결측치를 0으로 채우기
df = df.fillna(0)

# 'Cabin' 컬럼의 결측치를 'Unknown'이라는 문자로 채우기
df['Cabin'] = df['Cabin'].fillna('Unknown')
```

### ② 통계값으로 채우기 (평균, 중앙값, 최빈값) ⭐

가장 많이 쓰는 패턴입니다.

```python
# 1. 평균(mean)으로 채우기 (수치형)
mean_val = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_val)

# 2. 중앙값(median)으로 채우기 (수치형 - 이상치에 강함)
median_val = df['Age'].median()
df['Age'] = df['Age'].fillna(median_val)

# 3. 최빈값(mode)으로 채우기 (범주형 - 예: 성별)
# mode()는 시리즈(리스트)를 반환하므로 [0]을 꼭 붙여야 함!
mode_val = df['Gender'].mode()[0]
df['Gender'] = df['Gender'].fillna(mode_val)
```

### ③ 앞/뒤 값으로 채우기 (시계열 데이터용)

날짜 순서가 중요한 데이터에서 사용합니다.

```python
# 바로 앞의 값으로 채우기 (Forward Fill)
df = df.fillna(method='ffill')

# 바로 뒤의 값으로 채우기 (Backward Fill)
df = df.fillna(method='bfill')
```

-----

# ⚡️ 시험장 요약 노트 (Cheat Sheet)

1.  **확인:** `df.isnull().sum()`
2.  **삭제:** `df.dropna()`
3.  **채우기 (수치형):** `df['col'].fillna(df['col'].median())` (중앙값 추천)
4.  **채우기 (범주형):** `df['col'].fillna(df['col'].mode()[0])` (최빈값, `[0]` 주의\!)

이 4가지만 자유자재로 쓰시면 결측치 문제는 완벽합니다\!

빅데이터 분석기사 실기 3유형에서 \*\*F-검정(F-test)\*\*은 크게 **두 가지 상황**에서 등장합니다.

1.  **두 집단의 분산이 같은지 확인할 때** (등분산 검정)
2.  **세 집단 이상의 평균을 비교할 때** (ANOVA)

시험장용으로 딱 정리해 드립니다.

-----

### 1\. ⚖️ 등분산 검정 (t-검정의 사전 단계)

t-검정을 하기 전에 \*\*"두 그룹의 퍼짐 정도(분산)가 같은가?"\*\*를 확인할 때 씁니다.

  * **함수:** **`scipy.stats.levene`** (레빈 검정 - 가장 많이 씀)
      * (참고: `bartlett`도 있지만 데이터가 정규성을 만족해야 해서 `levene`이 더 안전합니다.)
  * **가설:**
      * $H_0$: 두 집단의 분산은 같다. (등분산)
      * $H_1$: 두 집단의 분산은 다르다. (이분산)

**💻 코드:**

```python
from scipy import stats

# A반, B반 점수 데이터
stat, p_val = stats.levene(group_A, group_B)

print(p_val)

# 판결 (t-검정 옵션 결정)
if p_val > 0.05:
    print("등분산 가정 만족 -> ttest_ind(..., equal_var=True)")
else:
    print("등분산 가정 기각 -> ttest_ind(..., equal_var=False)")
```

-----

### 2\. 📊 분산분석 (ANOVA)

세 개 이상의 그룹 평균을 비교할 때 사용하는 검정 방법 자체가 **F-검정**입니다.

  * **함수:** **`scipy.stats.f_oneway`**
  * **의미:** `(집단 간 분산) / (집단 내 분산)` 비율을 봅니다.
  * **가설:**
      * $H_0$: 모든 그룹의 평균이 같다.
      * $H_1$: 적어도 하나는 다르다.

**💻 코드:**

```python
from scipy import stats

# A, B, C 세 그룹 비교
f_stat, p_val = stats.f_oneway(group_A, group_B, group_C)

print(f"F-통계량: {f_stat}") # 이게 바로 F값
print(f"p-value: {p_val}")
```

-----

### 3\. 🧮 (심화) 수동 계산: 두 모분산의 비 검정

가끔 3유형 문제에서 \*\*"두 집단의 분산 비율(F값)을 직접 구하시오"\*\*라고 할 때가 있습니다. (방금 푸신 문제)

이때는 함수가 따로 없어서 **공식**을 써야 합니다.

  * **공식:** $F = \frac{\text{큰 분산}}{\text{작은 분산}}$
  * **주의:** 분산 구할 때 반드시 **`ddof=1`** (표본분산)을 써야 합니다.

**💻 코드:**

```python
import numpy as np

var_A = np.var(group_A, ddof=1)
var_B = np.var(group_B, ddof=1)

# 큰 게 위로 가도록
f_stat = var_A / var_B if var_A > var_B else var_B / var_A

print(f"F-통계량: {f_stat}")
```

-----

### ⚡️ 시험장 요약 노트

1.  **"t-검정 전에 분산 확인해라"** 👉 **`stats.levene`**
2.  **"세 그룹 평균 비교해라"** 👉 **`stats.f_oneway`**
3.  **"F-통계량을 직접 계산해라"** 👉 **`var / var`** (단, `ddof=1`)