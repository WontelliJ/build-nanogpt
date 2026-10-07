# LLM 동작 원리 (문장 입력부터 다음 Token 생성까지)

## 요약

- LLM은 입력 문장을 숫자 벡터로 변환하고 Transformer에서 Token 간 정보를 교환한 뒤 다음 Token 1개를 선택하는 과정을 반복함
- 각 단계는 직전 단계에서 남은 한계를 해결함
- 설명 대상: GPT, Claude, Llama 등 현재 대부분의 LLM이 사용하는 Decoder-only Transformer 구조
- 코드 참조: build-nanogpt의 GPT-2 구현. 구조가 공개되어 있으며 최근 LLM과 기본 골격이 같음
- 문서 내 Token ID, 벡터, 가중치 수치는 설명용 임의 값

<sub>LLM(Large Language Model, 대규모 언어 모델): 대량의 텍스트로 학습해 다음 단어를 예측하는 신경망 모델<br>Transformer(트랜스포머): 2017년 발표된 신경망 구조. 현재 LLM의 기반<br>Decoder-only: Transformer 중 문장 생성 부분만 사용하는 구조. GPT 계열 대부분이 해당</sub>

---

## 0. 예시 대화

```text
사용자: 민수는 사과를 좋아해.
AI:     민수는 사과를 좋아하는군요.

사용자: 오늘은 과일 가게에 갔어.
AI:     과일을 고르기 좋은 곳이네요.

사용자: 그럼 그 사람에게 뭘 사주면 좋을까?
AI:     사과를 추천해요.
```

- 마지막 질문에 민수, 사과 모두 없음
- 모델은 그 사람 = 민수, 민수가 좋아하는 것 = 사과를 연결해 답변 생성
- 이 문서는 이 연결이 만들어지는 계산 과정을 단계별로 설명

### 전체 흐름

| 단계 | 입력 → 출력 | 해결하는 한계 |
|---|---|---|
| 1. Tokenizer | 문장 → 정수 ID 목록 | 컴퓨터는 글자를 직접 계산할 수 없음 |
| 2. Embedding | ID → 벡터 | ID 번호에는 의미 정보가 없음 |
| 3. Position | 벡터 → 위치 정보가 더해진 벡터 | 순서 정보가 없음 |
| 5. Self-Attention | 벡터 → 다른 Token 정보가 반영된 벡터 | Token끼리 정보를 주고받지 못함 |
| 6. Multi-Head | Attention 1개 → 여러 개 | 한 번에 한 종류의 관계만 참조 가능 |
| 7. FFN | 벡터 → 변환된 벡터 | 가져온 정보가 섞인 상태로 남음 |
| 8. Residual, LayerNorm | 원래 벡터 + 변화량 | Layer가 깊어지면 정보 손실, 학습 불안정 |
| 9. Layer 반복 | Block × N | 한 번으로는 여러 단계 거친 관계 연결 불가 |
| 10. 출력 | 벡터 → Token별 확률 | 최종적으로 Token 1개를 골라야 함 |
| 11. 생성 반복 | 선택한 Token을 붙여 재계산 | 답변은 여러 Token으로 구성됨 |

4장은 3장과 5장의 순서가 정해진 배경(RNN → Transformer) 설명

---

## 1. Tokenizer

> 문장을 Token 단위로 자르고 각 Token에 사전 번호(ID)를 부여

```text
입력:   민수는 사과를 좋아해.

Token:  [민수] [는] [사과] [를] [좋아해] [.]
ID:     [ 12 ] [ 7] [ 31 ] [ 8] [  45  ] [2]
```

- 대화 전체는 역할 구분용 특수 Token과 함께 하나의 ID 목록으로 이어 붙여 입력됨

```text
<사용자> 민수는 사과를 좋아해. <AI> 민수는 ... <사용자> 그럼 그 사람에게 뭘 사주면 좋을까? <AI>
→ [900, 12, 7, 31, 8, 45, 2, 901, 12, 7, ..., 900, 88, 103, 57, 19, 64, 77, 5, 901]
```

- 변화: 문자열 → 정수 목록
- 남은 한계: ID는 사전 내 위치 번호일 뿐 의미 정보 없음

실제 LLM
- 단어 단위 대신 자주 함께 등장하는 글자 조합 단위로 분리 (BPE 계열)
- 사전 크기: GPT-2 50,257개 / GPT-4 약 10만 개 / Llama 3 128,256개
- 영어 위주로 만든 Tokenizer는 한글을 더 잘게 분리 → 같은 내용도 Token 수 증가

<sub>Token(토큰): 모델이 처리하는 텍스트 조각. 단어 또는 단어의 일부<br>Tokenizer(토크나이저): 문장을 Token으로 자르고 ID로 바꾸는 도구<br>Vocabulary(어휘 사전): 모델이 아는 Token 전체 목록<br>BPE(Byte Pair Encoding): 자주 붙어 나오는 글자 쌍을 반복해서 합쳐 Token 사전을 만드는 방식</sub>

---

## 2. Embedding

> ID를 실수 여러 개로 구성된 벡터로 변환. Token 간 유사도 계산이 가능해짐

### 변환 전 (ID)

```text
민수   → 12
사과   → 31
바나나 → 9137
```

- ID 차이 기준으로는 사과(31)가 바나나(9137)보다 민수(12)에 가까움
- ID 간 거리는 의미와 무관 → 유사도 계산 불가

### 변환 후 (벡터)

- Embedding 표에서 ID 번호에 해당하는 행을 조회
- 아래는 3차원으로 축소한 설명용 예

```text
사과   → [0.9, 0.8, 0.1]
바나나 → [0.8, 0.9, 0.2]
민수   → [0.1, 0.2, 0.9]

사과 · 바나나 = 0.9×0.8 + 0.8×0.9 + 0.1×0.2 = 1.46   (유사)
사과 · 민수   = 0.9×0.1 + 0.8×0.2 + 0.1×0.9 = 0.34   (비유사)
```

| 구분 | ID | 벡터 |
|---|---|---|
| 형태 | 정수 1개 | 실수 여러 개 |
| 유사도 계산 | 불가 | 내적으로 가능 |
| 값 결정 방식 | 사전 순서로 고정 | 학습으로 조정 |

- 각 차원의 의미는 사람이 지정하지 않음. 학습 과정에서 비슷하게 쓰이는 Token끼리 가까워지도록 값이 조정됨
- 이후 Attention도 이 내적 계산을 사용
- 벡터 차원 수: GPT-2 small 768 / GPT-3 12,288 / Llama 3 8B 4,096

### 남은 한계

- 문맥 미반영: 배가 고파서 사과를 먹었다(과일), 약속에 늦어서 사과를 했다(사죄)의 사과가 같은 벡터
- 순서 미반영: 문장 내 위치 정보 없음

<sub>벡터: 여러 개의 숫자를 순서대로 나열한 값<br>Embedding(임베딩): Token ID를 벡터로 바꾸는 조회 표. 표의 값도 학습으로 결정<br>내적: 두 벡터의 같은 위치 값끼리 곱해서 더한 값. 클수록 두 벡터의 방향이 비슷함</sub>

---

## 3. Position

> Token 벡터에 위치 정보를 더해서 같은 Token도 위치가 다르면 다른 벡터가 되도록 함

### 반영 전

```text
문장 A: 사람이 사과를 먹는다.  →  [사람] [이] [사과] [를] [먹는다]
문장 B: 사과가 사람을 먹는다.  →  [사과] [가] [사람] [을] [먹는다]
```

- Transformer는 Token을 앞에서부터 하나씩 읽지 않고 전체를 한 번에 입력받음 (이유는 4장)
- 위치 정보가 없으면 두 문장이 같은 Token 집합으로 처리됨

```text
{ 사람, 사과, 먹는다, 주격 조사(이/가), 목적격 조사(을/를) }
```

- 주격 조사가 사람에 붙었는지 사과에 붙었는지 판단 불가 → 두 문장 구분 불가

### 반영 후

```text
Transformer 입력 = Token 벡터 + 위치 벡터

문장 A: 사람+위치0, 이+위치1, 사과+위치2, 를+위치3, 먹는다+위치4
문장 B: 사과+위치0, 가+위치1, 사람+위치2, 을+위치3, 먹는다+위치4
```

- 0번 위치의 사과와 2번 위치의 사과가 서로 다른 벡터가 됨
- 주격 조사가 바로 앞 Token에 붙는다는 관계 계산 가능 → 두 문장 구분 가능

### 남은 한계

- 각 Token 벡터는 자기 정보만 보유
- 마지막 질문의 사람이 누구를 가리키는지 알 수 없음

실제 LLM
- 원 논문(2017): sin, cos 함수로 위치 벡터 계산
- GPT-2, GPT-3: 위치별 벡터를 학습으로 결정
- Llama, Qwen, Mistral 등 최근 공개 모델 다수: RoPE 사용. 더하는 대신 Q, K 벡터를 위치만큼 회전시켜 Token 간 상대 거리 반영. 학습 때보다 긴 입력으로 확장하기 쉬움

<sub>RoPE(Rotary Position Embedding, 회전 위치 임베딩): 위치에 비례하는 각도로 벡터를 회전시켜 위치 정보를 넣는 방식. Q, K는 5장 참조</sub>

---

## 4. RNN과 Transformer

> Transformer는 거리가 멀수록 문맥이 약해지는 RNN의 한계를 Attention으로 해결했고 그 과정에서 사라진 순서 정보를 입력 단계의 Position으로 보완함

### Transformer 이전의 RNN

```text
민수 → 는 → 사과 → 를 → 좋아해 → ... → 그 → 사람 → 에게 → 뭘 → 사주면
상태 → 상태 → 상태 → ... (매 단계 이전 상태를 갱신해 전달) ... → 상태
```

- Token을 앞에서부터 하나씩 처리하며 이전 내용을 고정 크기 상태값 하나에 누적해 전달
- 순서 정보는 처리 순서로 자연스럽게 반영됨
- 한계 1: 입력이 길수록 앞부분 정보가 약해짐. 예시에서 사주면까지 처리하는 시점에는 첫 문장의 민수, 사과 정보가 약해진 상태
- 한계 2: 앞 Token 처리가 끝나야 다음 Token 처리 가능 → GPU 병렬 계산이 어려움 → 모델 규모 확대가 어려움

### Transformer (2017, Attention Is All You Need)

- RNN 제거
- 각 Token이 다른 모든 Token을 직접 참조하고 참조 비율을 가중치로 계산 (Attention)
- 세 문장 앞의 사과도 바로 앞 Token과 같은 1단계로 참조 → 거리로 인한 정보 손실 해결
- 모든 Token을 동시에 계산 → 병렬 처리 가능
- 대신 순서 정보가 사라짐 → 입력 단계에서 위치 정보를 먼저 더함 (3장)

| 구분 | RNN | Attention만 사용 | Attention + Position |
|---|---|---|---|
| 먼 문맥 | 약해짐 | 유지 | 유지 |
| 순서 정보 | 있음 | 없음 | 있음 |
| 병렬 계산 | 어려움 | 가능 | 가능 |

### 이후 변화

- 병렬 계산이 가능해지면서 모델 크기와 학습 데이터를 크게 늘릴 수 있게 됨
- 모델 크기, 데이터, 계산량이 늘수록 성능이 일정하게 향상되는 경향 확인 (Scaling Law, 2020)

```text
2017  Transformer 발표 (번역 모델)
2018  GPT-1      1.17억 파라미터
2019  GPT-2      15억
2020  GPT-3      1,750억
2022  ChatGPT    대화형 서비스 출시
2023~ GPT-4, Claude, Llama, Gemini 등
```

- 기본 구조(Token → Embedding → Transformer Block 반복 → 다음 Token 예측)는 GPT-1부터 현재까지 동일
- 이후 변화는 주로 규모 확대와 세부 구성 요소 개선 (14장 비교표)

<sub>RNN(Recurrent Neural Network, 순환 신경망): 입력을 순서대로 하나씩 처리하며 이전 상태를 다음 단계로 넘기는 신경망<br>Attention(어텐션): 각 Token이 다른 Token을 얼마나 참고할지 가중치로 계산해 정보를 가져오는 연산<br>GPU: 대량의 계산을 동시에 처리하는 연산 장치<br>파라미터: 학습으로 정해지는 모델 내부의 숫자. 모델 크기의 기준<br>Scaling Law(스케일링 법칙): 규모가 커질수록 성능이 예측 가능한 비율로 향상된다는 관찰 결과</sub>

---

## 5. Self-Attention

> 각 Token이 차례로 Query가 되어 다른 Token들의 Key와 비교하고 관련도 비율만큼 Value를 가져와 자기 벡터에 더함

### Q, K, V 정의

| 구분 | 의미 | 예시 (사람 Token 기준) |
|---|---|---|
| Query | 현재 Token이 찾는 정보 | 가리키는 대상이 누구인가 |
| Key | 각 Token이 가진 정보의 특징. Query와 비교하는 데 사용 | 민수: 사람 이름 |
| Value | 선택되었을 때 실제로 전달되는 정보 | 민수에 관한 정보 |

- 세 값 모두 같은 Token 벡터에 서로 다른 가중치 행렬을 곱해 생성
- 행렬 값은 학습으로 결정

### 계산 순서

1. 한 Token이 Query가 됨
2. Query와 각 Token의 Key를 내적 → 관련도 점수
3. Softmax로 점수를 합이 1인 비율로 변환
4. 비율만큼 각 Token의 Value를 곱해 합산 → 현재 Token 벡터에 더함
5. 모든 Token이 Query가 되어 1~4 수행 (실제로는 행렬 연산으로 동시에 계산)

- 참조 범위: 자기 자신과 앞쪽 Token만 (Causal Attention)
- 이유: 학습 시 다음 Token이 정답이므로, 뒤쪽 Token을 참조하면 정답을 미리 보는 것과 같음
- 수식: softmax(Q·Kᵀ / √d) · V
- √d로 나누는 이유: 벡터 차원이 클수록 내적 값이 커져 Softmax 결과가 한 Token에 몰리는 현상 방지

<sub>Self-Attention(셀프 어텐션): 같은 입력 안의 Token끼리 수행하는 Attention<br>Query / Key / Value: 질의 / 키 / 값. 줄여서 Q, K, V<br>Softmax(소프트맥스): 점수 목록을 0~1 사이, 합이 1인 비율로 바꾸는 함수<br>Causal Attention(인과적 어텐션): 뒤쪽 Token을 가려 앞쪽만 참조하도록 제한한 Attention. 가리는 처리를 Causal Mask라고 함<br>d: Q, K 벡터의 차원 수</sub>

### 예시 1. 사람이 Query일 때

```text
Query(사람): 가리키는 대상이 누구인가

비교 대상 Key   점수(Q·K)   Softmax 비율
민수            5.0          0.70
그              3.2          0.12
과일 가게       2.8          0.08
사과            2.4          0.05
기타            ...          0.05

Attention 결과(사람) = 0.70×V(민수) + 0.12×V(그) + 0.08×V(과일 가게) + 0.05×V(사과) + ...
→ 원래 사람 벡터에 더해짐 (8장 Residual)
```

- 적용 전: 사람 = 대상이 정해지지 않은 일반 명사
- 적용 후: 사람 = 민수 정보가 70% 반영된 벡터 → 그 사람 = 민수 연결

### 예시 2. 사주면이 Query일 때

```text
Query(사주면): 무엇을 사 줘야 하는가

비교 대상 Key            Softmax 비율
사과                     0.55
좋아해                   0.20
사람 (민수 정보 반영됨)  0.15
기타                     0.10
```

- 적용 전: 사주면 = 무언가를 사 준다는 일반 의미
- 적용 후: 사주면 = 사과, 좋아함 정보가 반영된 벡터

### 2장 한계 해결

- 배가 고파서 사과를: 사과가 배가 고파서의 Value를 가져옴 → 과일 의미 벡터
- 약속에 늦어서 사과를: 사과가 약속에 늦어서의 Value를 가져옴 → 사죄 의미 벡터
- 같은 Token이라도 앞 문맥에 따라 다른 벡터가 됨

### 참조 비율 결정 기준

- 비율은 거리가 아니라 Query와 Key의 관련도로 결정
- 세 문장 앞의 민수, 사과도 높은 비율로 반영 가능

---

## 6. Multi-Head Attention

> Attention을 여러 개(Head)로 나눠 병렬 수행. 한 Token이 여러 종류의 관계를 동시에 참조

### Head가 1개일 때

- Softmax 비율 분포가 1개뿐
- 사주면이 대상(민수)과 대상이 좋아하는 것(사과)을 동시에 참조하려면 하나의 비율을 나눠 써야 함 → 각 정보가 약하게 반영

### Head가 여러 개일 때

```text
          ┌ Head 1: 대상이 누구인가        → 민수 비율 높음
사주면 ───┼ Head 2: 무엇을 원하는가        → 사과 비율 높음
          ├ Head 3: 문장 구조 (주어, 목적어) → 문법 관계
          └ ...
                   ↓ 결과를 이어 붙인 뒤 Linear
             사주면의 새 벡터
```

- 벡터 차원을 Head 수만큼 나눠 각 Head가 독립적으로 Q, K, V 계산
- 각 Head가 보는 관계는 학습으로 결정됨 (위 역할 구분은 설명용)
- Head 수: GPT-2 small 12개 (768 = 12 × 64) / GPT-3 96개 / Llama 3 8B 32개

실제 LLM
- 최근 모델 다수가 GQA 사용: 여러 Query Head가 Key, Value Head를 공유
- 생성 시 저장할 Key, Value 양이 줄어 속도와 메모리 사용량 개선 (Llama 3 8B: Query Head 32개, Key/Value Head 8개)

<sub>Head(헤드): Q, K, V 계산 1세트<br>Linear(선형 변환): 벡터에 가중치 행렬을 곱하는 연산<br>GQA(Grouped-Query Attention, 그룹 쿼리 어텐션): 여러 Query Head가 Key, Value를 나눠 쓰는 방식</sub>

---

## 7. FFN

> Attention은 다른 Token에서 정보를 가져오는 단계, FFN은 각 Token이 가져온 정보를 개별적으로 변환하는 단계

| 구분 | Attention | FFN |
|---|---|---|
| 다른 Token 참조 | 함 | 안 함 |
| 역할 | 정보 수집 | 수집한 정보 변환 |
| 연산 | Q·K 비교, Value 가중합 | Linear → 활성화 함수 → Linear |

- 예: Attention 이후 사주면 벡터에는 민수, 사과, 좋아함, 과일 가게 정보가 섞여 있음 → FFN이 다음 Layer에서 사용하기 좋은 형태로 변환

```text
GPT-2:    Linear(768 → 3072) → GELU → Linear(3072 → 768)
Llama 등: SwiGLU. Linear 2개의 결과를 곱한 뒤 Linear로 원래 차원 복원
```

- 파라미터 비중: Block 파라미터의 절반 이상 (GPT-2 약 2/3, Llama 3 8B 약 80%)
- 학습한 사실, 패턴의 상당 부분이 FFN 가중치에 저장된다는 분석 연구 있음

실제 LLM
- 일부 최신 모델(Mixtral, DeepSeek 등)은 MoE 사용
- FFN을 여러 개 두고 Token마다 일부만 계산 → 전체 파라미터는 늘리고 Token당 계산량은 유지

<sub>FFN(Feed-Forward Network, 피드포워드 신경망): Linear와 활성화 함수로 구성된 신경망. Token마다 독립적으로 적용<br>활성화 함수(GELU, SwiGLU 등): Linear 사이에 넣어 비선형 변환을 가능하게 하는 함수. 없으면 Linear를 여러 번 거쳐도 Linear 1번과 같은 결과<br>MoE(Mixture of Experts, 전문가 혼합): 여러 FFN 중 일부만 선택해 계산하는 구조</sub>

---

## 8. Residual Connection, LayerNorm

> Layer를 깊게 쌓아도 원래 정보가 유지되고 학습이 안정되도록 하는 구조

### Residual Connection

```text
x = x + Attention(Norm(x))
x = x + FFN(Norm(x))
```

- 적용 전: 각 단계가 벡터를 새로 생성 → Layer를 거칠수록 Token 원래 정보 손실, 앞쪽 Layer까지 학습 신호 전달 약화
- 적용 후: 기존 벡터를 유지하고 각 단계의 결과만 더함 → 원래 정보 보존, 수십~100여 개 Layer 학습 가능

### LayerNorm

- 각 Token 벡터 값의 평균과 크기를 일정 범위로 조정
- Layer를 거치며 값이 지나치게 커지거나 작아지는 현상 방지

실제 LLM
- 최근 모델 다수는 계산이 더 단순한 RMSNorm 사용 (Llama 등)

### Transformer Block 1개 구성

```text
입력 x
 → Norm → Attention → x에 더함
 → Norm → FFN       → x에 더함
출력 x
```

<sub>Residual Connection(잔차 연결): 입력을 출력에 그대로 더하는 연결<br>LayerNorm(Layer Normalization, 층 정규화): 벡터 값의 평균과 크기를 맞추는 연산<br>RMSNorm(Root Mean Square Normalization): 평균 조정 없이 크기만 맞추는 정규화<br>Block(블록): Attention, FFN, Residual, Norm을 묶은 반복 단위. Layer와 같은 의미로 사용</sub>

---

## 9. Layer 반복

> 같은 구조의 Block을 수십 층 반복. Layer마다 Value가 전달, 누적되며 여러 단계를 거친 관계 연결이 가능해짐

- Layer 수: GPT-2 small 12 / GPT-3 96 / Llama 3 8B 32 / Llama 3 70B 80

### Block 1개일 때

- 직접 연결된 Token 정보만 1회 참조
- 마지막 위치가 사람을 참조해도, 그 시점의 사람 벡터에 민수 정보가 아직 없을 수 있음

### 여러 층일 때

```text
Layer 1   사람     ← 민수 정보 반영                        (그 사람 = 민수)
Layer 2   사주면   ← 사람(민수 정보 포함), 사과 정보 반영   (민수에게 사과)
Layer 3~  마지막 위치 ← 앞 결과가 반영된 Token들 참조
...
Layer N   마지막 위치의 최종 벡터
```

- 정보 전달 경로: 사과 → 민수 → 그 사람 → 마지막 위치
- 층별 역할은 설명용. 실제로는 여러 Head, Layer에 분산됨
- Token 자체는 바뀌지 않음. 같은 위치의 벡터가 Layer마다 갱신됨

<sub>Hidden State(은닉 상태): 모델 내부의 벡터. 보통 마지막 Layer의 출력을 지칭</sub>

---

## 10. 출력

> 마지막 위치의 Hidden State를 사전 전체 Token에 대한 점수로 바꾸고 Softmax로 확률화한 뒤 1개 선택

- 마지막 위치만 사용하는 이유: Causal Attention으로 앞쪽 모든 Token의 정보가 마지막 위치에 반영되어 있음

```text
마지막 위치 Hidden State (d차원)
   ↓ Norm → lm_head (d → 사전 크기)
Logits (사전의 모든 Token별 점수)
   사과    8.4
   바나나  5.1
   포도    4.0
   꽃      3.2
   ...
   ↓ Softmax (위 4개만으로 계산한 예)
확률: 사과 95%, 바나나 3.5%, 포도 1.2%, 꽃 0.5%
   ↓ 선택
사과
```

선택 방식
- Greedy: 확률이 가장 높은 Token 선택
- Sampling: 확률에 비례해 무작위 선택. 같은 질문에 답변이 달라지는 원인
- Temperature: 확률 분포의 쏠림 정도를 조정하는 값. 높을수록 다양한 Token 선택

- GPT-2 등 일부 모델은 lm_head와 Embedding 표가 같은 가중치를 공유

<sub>Logits(로짓): Softmax 적용 전의 원점수<br>lm_head(출력층): Hidden State를 사전 크기의 점수 벡터로 바꾸는 Linear<br>Greedy(탐욕적 선택), Sampling(샘플링), Temperature(온도)</sub>

---

## 11. 생성 반복

> LLM은 Token을 1개씩 생성. 생성한 Token을 입력 뒤에 붙여 다음 Token을 다시 계산

```text
그럼 그 사람에게 뭘 사주면 좋을까?
                    ↓
                  사과
```

사과가 포함된 문맥으로 다음 Token 계산

```text
... 뭘 사주면 좋을까? 사과
                    ↓
                    를
                    ↓
                 추천해요
```

```text
문맥 → 1~10단계 계산 → 다음 Token 1개 → 문맥 뒤에 추가 → 다시 계산 → ... → 종료 Token
```

- 종료 조건: 종료 Token 생성 또는 최대 길이 도달

실제 LLM
- KV Cache 사용: 앞쪽 Token의 Key, Value를 저장해 두고 새 Token 계산만 추가 → 매번 전체를 다시 계산하지 않음
- Context Window: GPT-2 1,024 Token / 최근 LLM 128,000 Token 이상

<sub>KV Cache(키-값 캐시): 이전 Token의 Key, Value를 저장하는 공간<br>Context Window(컨텍스트 윈도우): 모델이 한 번에 참조할 수 있는 최대 Token 수</sub>

---

## 12. 학습

> Embedding, Q·K·V 행렬, FFN 등 모든 가중치는 무작위 값에서 시작. 다음 Token 예측 오차를 줄이는 방향으로 반복 수정

### 사전학습

```text
입력: 민수는 사과를     정답: 좋아해
  ↓ 모델 예측
예측 확률과 정답 비교 → Loss
  ↓ Backpropagation
모든 가중치를 정답 확률이 높아지는 방향으로 수정
  ↓
수조 개 Token 규모의 텍스트로 반복
```

- 다음 Token을 맞히는 과정에서 문법, 지시 대상, 사실 관계 등의 패턴이 가중치에 반영됨

### 사후학습 (대화형 LLM)

- 사전학습만 한 모델은 문장 이어쓰기만 가능. 질문에 답하는 형식은 추가 학습 필요
- SFT: 질문-답변 예시 데이터로 추가 학습
- RLHF: 사람이 더 낫다고 평가한 답변이 나오도록 추가 학습
- ChatGPT, Claude 등 대화형 서비스는 사전학습 + 사후학습을 거친 모델

| 구분 | Training | Inference |
|---|---|---|
| 목적 | 가중치 조정 | 고정된 가중치로 답변 생성 |
| 흐름 | 예측 → Loss → 역전파 → 수정 | 예측 → 선택 → 추가 → 반복 |

<sub>Loss(손실): 예측과 정답의 차이를 나타내는 값<br>Backpropagation(역전파): Loss를 기준으로 각 가중치의 수정 방향을 계산하는 알고리즘<br>Pre-training(사전학습) / Post-training(사후학습)<br>SFT(Supervised Fine-Tuning, 지도 미세조정)<br>RLHF(Reinforcement Learning from Human Feedback, 인간 피드백 기반 강화학습)<br>Training(학습) / Inference(추론): 가중치를 만드는 과정 / 만든 가중치로 결과를 생성하는 과정</sub>

---

## 13. 단계별 데이터 변화

T: Token 수, d: 벡터 차원, V: 사전 크기

| 단계 | 데이터 | 이 단계가 없을 때 |
|---|---|---|
| 입력 | 문자열 | - |
| Tokenizer | 정수 T개 | 계산 불가 |
| Embedding | T × d | Token 간 유사도 계산 불가 |
| Position | T × d | 순서 구분 불가 |
| Self-Attention (Multi-Head) | T × d | Token 간 정보 교환 불가 |
| FFN | T × d | 수집한 정보가 변환되지 않음 |
| Residual, Norm | T × d | 깊은 Layer에서 정보 손실, 학습 불안정 |
| Block × N | T × d | 여러 단계를 거친 관계 연결 불가 |
| lm_head | 1 × V (마지막 위치) | Token으로 변환 불가 |
| Softmax, 선택 | Token 1개 | - |
| 생성 반복 | 문장 | Token 1개로 종료 |

- Transformer: 각 Token이 Query로 필요한 정보를 찾고 Key로 대상을 비교한 뒤 Value를 가져와 자기 벡터에 누적하는 과정을 여러 Layer에서 반복하는 구조
- LLM: Transformer의 마지막 위치 벡터로 다음 Token을 1개씩 선택하는 과정을 반복

---

## 14. GPT-2와 최근 LLM 비교

| 항목 | GPT-2 small (2019) | GPT-3 (2020) | Llama 3 8B (2024) |
|---|---|---|---|
| 파라미터 | 1.24억 | 1,750억 | 80억 |
| Layer 수 | 12 | 96 | 32 |
| 벡터 차원 | 768 | 12,288 | 4,096 |
| Head 수 | 12 | 96 | Query 32 / Key·Value 8 |
| 사전 크기 | 50,257 | 50,257 | 128,256 |
| Context Window | 1,024 | 2,048 | 8,192 (3.1 버전: 128K) |
| 위치 정보 | 학습형 위치 벡터 | 학습형 위치 벡터 | RoPE |
| 정규화 | LayerNorm | LayerNorm | RMSNorm |
| FFN 활성화 함수 | GELU | GELU | SwiGLU |

- GPT-4 이후 OpenAI 모델, Claude 등은 구조 세부를 공개하지 않음
- 공개 모델 기준으로 기본 골격(Decoder-only, Causal Self-Attention, FFN, Residual, Block 반복, 다음 Token 예측)은 동일

---

## 15. 코드 대응 (build-nanogpt/train_gpt2.py)

| 개념 | 위치 | 코드 |
|---|---|---|
| Embedding, 위치 벡터 | GPT.\_\_init\_\_ L86–87 | `wte = nn.Embedding(vocab_size, n_embd)`, `wpe = nn.Embedding(block_size, n_embd)` |
| Token + 위치 | GPT.forward L116–118 | `x = tok_emb + pos_emb` |
| Q, K, V 생성 | CausalSelfAttention L31–32 | `qkv = self.c_attn(x)`, `q, k, v = qkv.split(...)` |
| Head 분할 | L33–35 | `.view(B, T, n_head, C // n_head)` |
| Causal Attention | L36 | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| FFN | MLP L46–48 | `Linear(768→3072)` → `GELU` → `Linear(3072→768)` |
| Residual + LayerNorm | Block.forward L67–68 | `x = x + self.attn(self.ln_1(x))`, `x = x + self.mlp(self.ln_2(x))` |
| Layer 반복 | GPT.forward L120–121 | `for block in self.transformer.h: x = block(x)` |
| Logits | L123–124 | `ln_f` → `lm_head` |
| 가중치 공유 | L94 | `wte.weight = lm_head.weight` |
| 생성 반복 | L461–473 | 마지막 위치 logits → softmax → top-k 샘플링 → `torch.cat`으로 추가 |
| 설정값 | GPTConfig L72–77 | `vocab_size=50257, n_layer=12, n_head=12, n_embd=768` |

---

## References

1. Vaswani et al., Attention Is All You Need, 2017. https://arxiv.org/abs/1706.03762
2. Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2), 2019
3. Brown et al., Language Models are Few-Shot Learners (GPT-3), 2020. https://arxiv.org/abs/2005.14165
4. Kaplan et al., Scaling Laws for Neural Language Models, 2020. https://arxiv.org/abs/2001.08361
5. Ouyang et al., Training language models to follow instructions with human feedback, 2022. https://arxiv.org/abs/2203.02155
6. Llama Team, The Llama 3 Herd of Models, 2024. https://arxiv.org/abs/2407.21783
7. Andrej Karpathy, build-nanogpt. https://github.com/karpathy/build-nanogpt
