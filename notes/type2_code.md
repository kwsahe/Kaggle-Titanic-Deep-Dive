빅데이터분석기사 실기 시험에서 사용하는 `sklearn.metrics`의 핵심 평가지표들을 코드로 정리했습니다. 시험장에서는 이 코드들을 복사하거나 외워서 사용하시면 됩니다.

-----

### ⚡️ [Cheatsheet] 필수 Import 모음 (복붙용)

```python
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_log_error # RMSLE용
from sklearn.metrics import confusion_matrix, classification_report
```

-----

### 1️⃣ 분류 (Classification) 평가지표

  * **주의 1:** `predict()` 결과값(0, 1 클래스)을 넣는 것과 `predict_proba()` 결과값(확률)을 넣는 것을 구분해야 합니다.
  * **주의 2:** 다중분류(3개 이상 클래스)에서는 `average='macro'` 옵션이 필수입니다.

<!-- end list -->

```python
# 가상의 정답(y_true)과 예측값(y_pred)
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]       # 모델.predict() 결과
y_prob = [0.1, 0.9, 0.4, 0.2, 0.8] # 모델.predict_proba()[:, 1] 결과

# 1. 정확도 (Accuracy)
acc = accuracy_score(y_true, y_pred)

# 2. 정밀도 (Precision) - 양성이라 예측한 것 중 실제 양성
prec = precision_score(y_true, y_pred)

# 3. 재현율 (Recall) - 실제 양성 중 양성이라 예측한 것
rec = recall_score(y_true, y_pred)

# 4. F1-Score (정밀도와 재현율의 조화평균) ★ 자주 출제
f1 = f1_score(y_true, y_pred)

# 5. ROC-AUC ★ 확률(Probability) 사용 필수!
auc = roc_auc_score(y_true, y_prob) 

print(f"정확도: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

# ---------------------------------------------------------
# [심화] 다중 분류 (Multi-class)일 때 (예: 0, 1, 2 예측)
# 반드시 average='macro' (또는 'micro', 'weighted')를 써야 함
# ---------------------------------------------------------
# f1_macro = f1_score(y_true_multi, y_pred_multi, average='macro')
# prec_macro = precision_score(y_true_multi, y_pred_multi, average='macro')
```

-----

### 2️⃣ 회귀 (Regression) 평가지표

  * **주의:** 회귀는 오차(Error)이므로 값이 **작을수록** 좋은 모델입니다. (R2 Score만 클수록 좋음)
  * **RMSE**는 별도 함수가 없어서 MSE에 루트(`np.sqrt`)를 씌워야 합니다.

<!-- end list -->

```python
# 가상의 정답(y_true)과 예측값(y_pred)
y_true = [100, 200, 300, 400]
y_pred = [110, 190, 320, 380]

# 1. MAE (Mean Absolute Error) - 평균 절대 오차
mae = mean_absolute_error(y_true, y_pred)

# 2. MSE (Mean Squared Error) - 평균 제곱 오차
mse = mean_squared_error(y_true, y_pred)

# 3. RMSE (Root Mean Squared Error) - MSE에 루트 씌움 ★ 자주 출제
# (방법 A: 정석)
rmse = np.sqrt(mse)
# (방법 B: 최신 버전 sklearn)
# rmse = mean_squared_error(y_true, y_pred, squared=False) 

# 4. R2 Score (결정계수) - 1에 가까울수록 좋음
r2 = r2_score(y_true, y_pred)

# 5. RMSLE (Root Mean Squared Log Error) - 로그변환 후 RMSE
# (값의 범위가 크거나, 0에 가까운 값이 중요할 때 사용)
# 주의: 음수 값이 있으면 에러남
msle = mean_squared_log_error(y_true, y_pred)
rmsle = np.sqrt(msle)

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
```

-----

### 📝 시험장 꿀팁 (이것만은 꼭\!)

1.  **ROC-AUC는 무조건 `predict_proba`**:

      * `model.predict(X_test)` ➡️ (X) 에러는 안 나지만 점수가 낮게 나옴.
      * `model.predict_proba(X_test)[:, 1]` ➡️ (O) 1일 확률을 넣어야 함.

2.  **다중 분류 F1-Score는 `average='macro'`**:

      * 시험 문제에서 "평가지표: Macro F1"이라고 명시되면 `f1_score(..., average='macro')`를 안 쓰면 에러가 납니다.

3.  **RMSE 계산**:

      * `sklearn` 버전에 따라 `squared=False` 옵션이 안 먹힐 수도 있습니다.
      * 안전하게 **`np.sqrt(mean_squared_error(y, pred))`** 방식을 외우는 게 좋습니다.

      `[:, 1]`은 파이썬(NumPy)에서 \*\*"2차원 표(행렬)를 자르는 문법(Slicing)"\*\*입니다.

한마디로 정의하면: \*\*"모든 줄(행)을 다 가져오되, 두 번째 칸(열)만 뽑아라"\*\*라는 뜻입니다.

-----

### 🔍 해부학적 분석

`[ : , 1 ]`

1.  **`:` (콜론)**

      * **의미:** "처음부터 끝까지 전부 다"
      * **위치:** 쉼표 앞(행 자리)에 있으므로 \*\*"모든 사람(데이터)을 다 선택해라"\*\*는 뜻입니다.

2.  **`,` (쉼표)**

      * **의미:** 행(가로)과 열(세로)을 나누는 기준선입니다.

3.  **`1` (숫자 1)**

      * **의미:** **"인덱스 1번 컬럼"** (즉, 두 번째 컬럼)
      * 파이썬은 숫자를 0부터 셉니다.
          * `0`번 컬럼: 첫 번째 (Class 0, 실패/사망 확률)
          * `1`번 컬럼: 두 번째 (Class 1, 성공/생존 확률)

-----

### 💡 그림으로 이해하기 (`predict_proba`의 결과물)

모델이 `predict_proba(X)`를 실행하면 아래와 같은 **2줄짜리 표**를 뱉어냅니다.

| (인덱스) | **[0]번 컬럼**<br>(안 살 확률) | **[1]번 컬럼**<br>(살 확률) |
| :--- | :--- | :--- |
| 고객 A | 0.8 (안 삼) | **0.2** (삼) |
| 고객 B | 0.1 (안 삼) | **0.9** (삼) |
| 고객 C | 0.4 (안 삼) | **0.6** (삼) |

여기서 우리는 \*\*"살 확률(Class 1)"\*\*만 필요합니다.

  * `predict_proba(X)` 전체 → 표 전체를 가져옴 (0.8, 0.2 둘 다)
  * `predict_proba(X)[:, 0]` → `[0.8, 0.1, 0.4]` (안 살 확률만 뽑음)
  * **`predict_proba(X)[:, 1]`** → **`[0.2, 0.9, 0.6]` (살 확률만 뽑음)** ✅ **(정답\!)**

-----

### 🚀 요약

시험에서 **ROC-AUC**나 **확률 제출** 문제가 나오면, 우리는 \*\*"정답(1)일 확률"\*\*이 필요합니다.

그래서 무조건 기계적으로 \*\*`[:, 1]`\*\*을 붙여서 \*\*오른쪽 줄(Class 1)\*\*만 쏙 빼오는 것입니다.

```python
# 모델이 뱉은 전체 확률 (2개 컬럼)
full_prob = [[0.9, 0.1], 
             [0.2, 0.8]]

# [:, 1]을 하면? -> 뒤에 있는 0.1, 0.8만 가져옴!
final_prob = [0.1, 0.8] 
```

이 코드는 빅데이터 분석기사 실기(2유형)에서 **가장 중요한 '필살기' 패턴**입니다.

\*\*"Train 데이터와 Test 데이터의 컬럼 개수가 달라서 생기는 에러"\*\*를 100% 예방하는 방법입니다. 단계별로 명확하게 정리해 드릴게요.

-----

### 1\. 왜 이렇게 해야 하나요? (이유)

만약 Train에는 \*\*'제주'\*\*라는 지역이 있는데, Test에는 \*\*'제주'\*\*가 없다면 어떻게 될까요?

  * **따로 인코딩 했을 때:**

      * `Train`: [서울, 부산, **제주**] (컬럼 3개)
      * `Test`: [서울, 부산] (컬럼 2개)
      * **결과:** 모델이 "어? 학습할 땐 컬럼이 3개였는데 왜 2개밖에 안 줘?" 하고 \*\*에러(Error)\*\*를 뿜습니다.

  * **합쳐서 인코딩 했을 때 (정답):**

      * `Combined`: [서울, 부산, 제주] (컬럼 3개로 통일)
      * **결과:** Test 데이터의 '제주' 컬럼은 모두 0으로 채워져서, **컬럼 개수가 완벽하게 일치**하게 됩니다.

-----

### 2\. 코드 상세 해석

#### ① 데이터 합치기 (`pd.concat`)

```python
# Train(X)과 Test(X_submit)를 위아래로 붙입니다.
combined = pd.concat([X, X_submit])
```

  * 이제 `combined` 변수 안에는 모든 범주(Category) 정보가 다 들어있습니다.

#### ② 원-핫 인코딩 한방에 수행 (`get_dummies`)

```python
# 합쳐진 상태에서 인코딩을 하니까 컬럼이 똑같이 생성됩니다.
combined_encoded = pd.get_dummies(combined, columns=['지역', '등급'])
```

#### ③ 다시 나누기 (`iloc`) - **가장 중요\!** ⭐

합쳐진 덩어리를 원래대로 **Train**과 **Test**로 잘라내야 합니다. 이때 \*\*"원래 Train의 데이터 개수(`len(X)`)"\*\*를 기준으로 자릅니다.

```python
# 1. 처음부터 ~ Train 개수만큼 자른다 -> 다시 Train(X)이 됨
X = combined_encoded.iloc[:len(X)]

# 2. Train 개수부터 ~ 끝까지 자른다 -> 다시 Test(X_submit)가 됨
X_submit = combined_encoded.iloc[len(X):]
```

-----

### ⚡️ 시험장 암기 요약

> **"원-핫 인코딩(`get_dummies`)을 할 때는 무조건 '합치고 ➡️ 바꾸고 ➡️ 나눈다'\!"**

이 3단계만 지키면 인코딩 관련 에러는 절대 나지 않습니다. 아주 훌륭한 코드 패턴입니다\!

# **📌 주요 차이점 요약(랜덤포레스트)**
| 항목              | 분류 (`RandomForestClassifier`) | 회귀 (`RandomForestRegressor`) |
| --------------- | ----------------------------- | ---------------------------- |
| 대상 값            | 범주형 (0,1,2 등)                 | 연속형 (실수값)                    |
| 평가 지표           | F1-score, Accuracy            | RMSE, MAE, R²                |
| `.predict()` 결과 | 정수 class                      | 실수값 예측                       |

# ***정석 코드***

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 데이터 불러오기
train = pd.read_csv('6_2_train.csv')
test = pd.read_csv('6_2_test.csv')

# 결측치 처리
train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
test['Gender'] = test['Gender'].fillna(train['Gender'].mode()[0])

# 데이터 분리
X = train.drop(columns=['ID', 'DBP'])
y = train['DBP']
X_test = test.drop(columns=['ID'])

# 인코딩
X = pd.get_dummies(X, columns=['Gender'])
X_test = pd.get_dummies(X_test, columns=['Gender'])
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 스플릿 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# 모델 학습 및 검증
model = RandomForestRegressor(random_state=0)
model.fit(X_train, y_train)
val_pred = model.predict(X_val)

# RMSE 평가
rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print("검증 RMSE:", round(rmse, 4))


# 전체 데이터로 재학습 후 예측
model.fit(X, y)
final_pred = model.predict(X_test)

# 결과 저장
result = pd.DataFrame({
    'ID': test['ID'],
    'pred': final_pred
})
result.to_csv('result.csv', index=False)
print(result.head())
```