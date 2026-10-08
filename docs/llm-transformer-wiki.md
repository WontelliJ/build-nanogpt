# LLM 동작 원리 (문장 입력부터 다음 Token 생성까지)

## 한눈에 보기

- LLM은 입력 문장을 숫자 벡터로 바꾼다. 그다음 Transformer 안에서 Token끼리 정보를 주고받게 한 뒤 그 결과로 다음에 올 Token 하나를 고른다.
- 답변이 끝날 때까지 이 과정을 반복한다. 긴 답변도 Token을 하나씩 이어 붙여 만든 것이다.
- GPT, Claude, Llama 같은 현재 대화형 LLM은 모두 이 구조를 기본으로 쓴다.
- 코드는 구조가 공개된 GPT-2 구현(이 저장소의 train_gpt2.py)을 참고했다.
- 문서 속 Token 분리, 번호, 벡터 값, 비율은 따라가기 쉽게 만든 설명용 값이다. 실제 Tokenizer로 돌려 본 결과는 1장에 따로 실었다.

<sub>LLM(Large Language Model, 대규모 언어 모델): 많은 양의 글을 학습해서 주어진 글 다음에 올 말을 예측하는 모델<br>
Token(토큰): 모델이 글을 처리하는 단위. 단어 하나일 수도 있고 글자 한두 개나 조사 하나일 수도 있다<br>
숫자 벡터: 숫자 여러 개를 한 줄로 묶은 값. 예: [0.9, 0.8, 0.1]. 모델은 글자 대신 이 숫자로 계산한다<br>
Transformer(트랜스포머): 2017년 발표된 신경망 구조. 문장 속 Token들이 서로의 정보를 참고하도록 만든다. 현재 LLM 대부분의 기반<br>
신경망: 입력 숫자에 가중치를 곱하고 더하는 계산을 여러 층 쌓은 모델<br>
가중치(파라미터): 신경망 안에서 곱해지는 숫자들. 처음엔 무작위 값이고 학습으로 정해진다. 모델 크기를 말할 때 이 숫자의 개수를 센다<br>
GPT(Generative Pre-trained Transformer): OpenAI가 만든 Transformer 기반 언어 모델 계열</sub>

---

## 0. 따라갈 예시

```text
사용자: 민수는 사과를 좋아해.
AI:     민수는 사과를 좋아하는군요.

사용자: 오늘은 과일 가게에 갔어.
AI:     과일을 고르기 좋은 곳이네요.

사용자: 그럼 그 사람에게 뭘 사주면 좋을까?
AI:     사과를 추천해요.
```

- 세 번째 질문에는 민수도 사과도 나오지 않는다. 그런데도 모델은 그 사람이 민수이고 민수가 사과를 좋아한다는 앞 대화 내용을 찾아 사과를 추천한다.
- 이 문서는 이 답이 나오기까지 데이터가 단계마다 어떤 모양으로 바뀌는지 따라간다.

---

## 전체 흐름

| 단계 | 하는 일 | 예시 (들어가는 것 → 나오는 것) | 이 단계가 없으면 |
|---|---|---|---|
| 1. Tokenizer | 문장을 Token으로 자르고 번호를 붙인다 | 민수는 사과를 좋아해. → [12, 7, 31, 8, 45, 2] | 컴퓨터가 글자를 계산할 수 없다 |
| 2. Embedding | 번호를 숫자 벡터로 바꾼다 | 31 → [0.9, 0.8, 0.1] | 사과와 바나나가 비슷하다는 것을 알 수 없다 |
| 3. 위치 정보 | 몇 번째 자리인지를 벡터에 더한다 | 사과 벡터 + 2번 자리 벡터 | 사람이 사과를 먹는다와 사과가 사람을 먹는다를 구분하지 못한다 |
| 4. Self-Attention | 각 Token이 앞쪽 Token에서 필요한 정보를 가져온다 | 그 사람 → 민수 정보가 섞인 그 사람 | 그 사람이 누구인지 알 수 없다 |
| 5. Multi-Head | 정보 가져오기를 여러 갈래로 나눠 동시에 한다 | 사주면 → 누구에게(민수)와 무엇을(사과)을 함께 가져옴 | 한 번에 한 가지 관계만 챙긴다 |
| 6. FFN | 가져온 정보를 Token마다 따로 가공한다 | 섞여 있는 벡터 → 다음 층에서 쓰기 좋은 벡터 | 가져온 정보가 뒤섞인 채로 남는다 |
| 7. Residual, LayerNorm | 원래 벡터에 새 정보를 더하고 값의 크기를 맞춘다 | 원래 벡터 + 새 정보 | 층을 많이 쌓으면 처음 정보가 사라지고 값이 불안정해진다 |
| 8. Layer 반복 | 4~7을 수십 번 반복한다 | 사과 → 민수 → 그 사람 → 마지막 자리 | 여러 번 건너야 이어지는 관계를 연결하지 못한다 |
| 9. 출력 | 마지막 자리 벡터로 다음 Token 확률을 계산해 하나 고른다 | 사과 95%, 바나나 3.5% … → 사과 | 계산 결과를 말로 바꿀 수 없다 |
| 10. 생성 반복 | 고른 Token을 붙이고 다시 계산한다 | … 좋을까? 사과 → 를 → 추천해요 | 답이 Token 하나에서 끝난다 |

<sub>Tokenizer(토크나이저): 문장을 Token으로 자르고 번호로 바꾸는 도구<br>
Token ID: 각 Token에 붙은 사전 번호<br>
Embedding(임베딩): Token ID를 숫자 벡터로 바꾸는 표. 번호로 줄을 찾아 그 줄의 숫자들을 꺼낸다<br>
위치 벡터: 0번 자리, 1번 자리처럼 자리마다 하나씩 정해 둔 숫자 벡터<br>
Self-Attention(셀프 어텐션): 같은 입력 안의 Token끼리 서로 얼마나 참고할지 계산해서 정보를 가져오는 연산<br>
Multi-Head(멀티 헤드): Self-Attention을 여러 갈래(Head)로 나눠 동시에 하는 방식<br>
FFN(Feed-Forward Network, 피드포워드 신경망): Token 하나의 벡터만 받아서 변환하는 작은 신경망<br>
Residual Connection(잔차 연결): 원래 벡터를 버리지 않고 새로 계산한 값을 더하는 연결<br>
LayerNorm(층 정규화): 벡터 값들의 평균과 크기를 일정하게 맞추는 계산<br>
Layer(층): 4~7단계를 묶은 계산 단위 하나. Block이라고도 한다<br>
확률: 후보 Token마다 다음에 올 가능성을 0~100%로 나타낸 값. 모두 더하면 100%</sub>

---

## 1. Tokenizer

> 문장을 Token 단위로 자르고 각 Token에 사전 번호(Token ID)를 붙인다.

### 예시 (설명용 분리)

```text
첫 번째 문장
  입력      민수는 사과를 좋아해.
  Token     [민수] [는] [사과] [를] [좋아해] [.]
  Token ID  [ 12 ] [ 7] [ 31 ] [ 8] [  45  ] [2]

세 번째 질문
  입력      그럼 그 사람에게 뭘 사주면 좋을까?
  Token     [그럼] [그] [사람] [에게] [뭘] [사주면] [좋을까] [?]
  Token ID  [ 88 ] [103] [ 57 ] [ 19 ] [64] [  77  ] [  91  ] [5]
```

- 따라가기 쉽도록 단어와 조사 단위로 잘랐다. 뒤에 나오는 사람, 사주면은 이 표의 Token을 가리킨다.
- 번호는 Vocabulary에서 그 Token이 놓인 자리다. 사과가 31번이라는 것은 사전의 31번째 칸이라는 뜻일 뿐이다.

<sub>Vocabulary(어휘 사전): 모델이 아는 Token 전체 목록. 각 Token은 목록 속 자리 번호를 ID로 가진다</sub>

### 실제 Tokenizer로 자른 결과

```text
GPT-4o Tokenizer (o200k_base)
  민수는 사과를 좋아해.   → 9개
  [민] [수] [는] [ 사] [과] [를] [ 좋아] [해] [.]
  ID: 24019, 7820, 2770, 9023, 8562, 4831, 90032, 5650, 13

  그럼 그 사람에게 뭘 사주면 좋을까?   → 15개
  [그] [럼] [ 그] [ 사람] [에게] … [ 사] [주] [면] [ 좋] [을] [까] [?]

GPT-2 Tokenizer
  Minsu likes apples.     → 6개   [M] [ins] [u] [ likes] [ apples] [.]
  민수는 사과를 좋아해.   → 26개  한글 한 글자가 2~3개 Token으로 쪼개짐
  apple → 1개 / 사과 → 5개
```

- 띄어쓰기도 Token에 들어간다. [ 사]는 공백과 사가 합쳐진 Token이다.
- 실제로는 사과, 사주면도 한 Token이 아니다. 원리는 같으므로 이 문서에서는 위의 단순한 분리로 설명한다.

### Tokenizer가 자르는 기준 (BPE)

> 많은 글에서 자주 붙어 나오는 글자 조합을 하나의 Token으로 등록한다.

```text
1. 처음에는 모든 글자(정확히는 바이트)를 따로 둔다        [좋] [아] [해]
2. 학습용 글 전체에서 가장 자주 붙어 나오는 쌍을 찾는다    [좋][아] 쌍이 가장 많다
3. 그 쌍을 하나로 합쳐 사전에 등록한다                     [좋아] [해]
4. 정해 둔 사전 크기가 될 때까지 2~3을 반복한다
```

- 이 방식이면 사전에 없는 단어도 처리할 수 있다. 처음 보는 사과나무숲도 아는 조각인 [사과] [나무] [숲]처럼 나눠서 받는다.
- 사전 크기는 GPT-2가 50,257개, GPT-4o가 약 20만 개다.

<sub>BPE(Byte Pair Encoding, 바이트 쌍 인코딩): 자주 붙어 나오는 글자 쌍을 반복해서 합쳐 Token 사전을 만드는 방법<br>
바이트: 컴퓨터가 글자를 저장하는 기본 단위. 영어 알파벳 한 글자는 1바이트, 한글 한 글자는 3바이트(UTF-8 기준)</sub>

### 한국어가 Token을 더 많이 쓰는 이유

- GPT-2 Tokenizer는 영어 글 위주로 만들어졌다. 영어 단어는 통째로 합쳐 두었지만 한글 조합은 거의 합치지 못했다.
- 그래서 한글 한 글자가 바이트 조각 2~3개로 나뉜다. 같은 뜻의 문장이 영어로는 6개, 한국어로는 26개였다.
- Token 수가 많으면 계산량과 사용료가 늘고 한 번에 넣을 수 있는 내용이 줄어든다.
- 최근 Tokenizer는 여러 언어를 함께 학습해서 이 차이가 줄었다. GPT-4o는 같은 한국어 문장을 9개로 자른다.

### 대화는 어떻게 들어가나

- 대화형 LLM은 여러 차례 주고받은 대화를 하나의 긴 Token 목록으로 이어 붙여 입력한다.
- 누가 한 말인지 구분하는 특수 Token을 함께 넣는다. 설명용으로 만든 말이 아니라 실제로 있는 Token이며 이름과 형식은 모델마다 다르다.

```text
Llama 3의 실제 대화 형식
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

민수는 사과를 좋아해.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

민수는 사과를 좋아하는군요.<|eot_id|> …
```

- <|start_header_id|>user<|end_header_id|>는 여기부터 사용자의 말이라는 표시다. <|eot_id|>는 한 사람의 말이 끝났다는 표시다.
- GPT-2처럼 대화 학습을 하지 않은 모델에는 이런 Token이 없다. 대화형 모델은 학습 마지막 단계에서 이 형식을 익힌다.

<sub>특수 Token: 글자가 아니라 역할을 나타내는 Token. 문장 시작, 화자 구분, 말 끝 같은 표시에 쓴다</sub>

### 이 단계에서 바뀐 것

- 문장이 정수 목록이 됐다. 이제 컴퓨터가 다룰 수 있다.
- 아직 남은 문제: 번호에는 뜻이 없다. 다음 단계에서 이 문제를 다룬다.

---

## 2. Embedding

> Token ID를 숫자 벡터로 바꾼다. 벡터끼리는 얼마나 비슷한지 계산할 수 있다.

### 바꾸기 전 (ID)

```text
민수   → 12
사과   → 31
바나나 → 9137
```

- 번호 차이로만 보면 사과(31)는 바나나(9137)보다 민수(12)에 훨씬 가깝다.
- 번호는 사전 속 자리일 뿐이라 이런 비교에는 의미가 없다. 사과와 바나나가 비슷하다는 사실을 번호로는 계산할 수 없다.

### 바꾼 후 (벡터)

- Embedding 표에는 사전의 Token 수만큼 줄이 있다. ID 31이 들어오면 31번째 줄의 숫자들을 꺼낸다.

```text
Embedding 표 (칸을 3개로 줄인 예)
  ID     벡터
  12     [0.1, 0.2, 0.9]    민수
  31     [0.9, 0.8, 0.1]    사과
  9137   [0.8, 0.9, 0.2]    바나나

내적 (같은 칸끼리 곱해서 모두 더하기)
  사과·바나나 = 0.9×0.8 + 0.8×0.9 + 0.1×0.2 = 1.46   큼 = 비슷함
  사과·민수   = 0.9×0.1 + 0.8×0.2 + 0.1×0.9 = 0.34   작음 = 다름
```

- 실제 벡터는 GPT-2가 768칸이고 최근 대형 모델은 수천 칸에서 1만 칸 이상이다.

<sub>내적: 두 벡터의 같은 칸끼리 곱한 뒤 모두 더한 값. 두 벡터가 같은 방향을 가리킬수록 커진다<br>
차원(칸 수): 벡터 하나에 들어 있는 숫자의 개수</sub>

### 벡터 값은 처음에 어떻게 정해지나

- 처음에는 작은 무작위 숫자로 채운다. GPT-2 코드는 평균 0, 표준편차 0.02인 무작위 값을 넣는다(train_gpt2.py 108번째 줄).
- 이 상태에서는 사과와 바나나가 전혀 비슷하지 않다.

```text
학습 전 (무작위)
  사과   [ 0.01, -0.03, 0.02]
  바나나 [-0.02,  0.01, 0.03]
  사과·바나나 = -0.0002 - 0.0003 + 0.0006 = 0.0001    관계 없음

학습 후
  사과   [0.9, 0.8, 0.1]
  바나나 [0.8, 0.9, 0.2]
  사과·바나나 = 1.46
```

- 학습은 다음 Token을 더 잘 맞히도록 숫자를 조금씩 고치는 과정이다.
- 사과를 먹었다, 바나나를 먹었다처럼 같은 자리에 자주 오는 Token은 비슷한 다음 Token을 예측하는 데 쓰인다. 그래서 고쳐지는 방향도 비슷해지고 두 벡터가 가까워진다.
- 칸마다 과일, 빨간색 같은 뜻이 정해져 있는 것은 아니다. 사람이 칸의 뜻을 정하지 않았고 여러 칸의 값이 함께 의미를 나타낸다.

<sub>표준편차: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 수. 0.02면 대부분 -0.04 ~ 0.04 사이의 작은 값이다</sub>

### 이 단계에서 바뀐 것

- Token 사이의 비슷함을 숫자로 계산할 수 있게 됐다. 뒤에 나오는 Attention도 이 내적 계산을 쓴다.
- 아직 남은 문제 1. 앞뒤 문맥을 모른다. 배가 고파서 사과를 먹었다의 사과(과일)와 약속에 늦어서 사과를 했다의 사과(사죄)가 같은 31번 줄에서 나온 같은 벡터다.
- 아직 남은 문제 2. 순서를 모른다. 벡터에는 이 Token이 문장의 몇 번째인지 들어 있지 않다.

---

## 3. 위치 정보

> Token 벡터에 몇 번째 자리인지를 나타내는 위치 벡터를 더한다. 같은 Token이라도 자리가 다르면 다른 벡터가 된다.

### 더하기 전

```text
문장 A: 사람이 사과를 먹는다.  →  [사람] [이] [사과] [를] [먹는다]
문장 B: 사과가 사람을 먹는다.  →  [사과] [가] [사람] [을] [먹는다]
```

- Transformer는 문장을 앞에서부터 한 단어씩 읽지 않는다. 모든 Token 벡터를 한꺼번에 받아 계산한다.
- 그래서 위치 정보가 없으면 두 문장 모두 아래와 같은 Token 묶음으로 보인다.

```text
{ 사람, 사과, 먹는다, 주격 조사(이/가), 목적격 조사(을/를) }
```

- 주격 조사가 사람 뒤에 붙었는지 사과 뒤에 붙었는지 알 수 없다. 누가 누구를 먹는지 구분하지 못한다.

<sub>주격 조사: 문장의 주어임을 나타내는 조사(이/가). 목적격 조사: 목적어임을 나타내는 조사(을/를)</sub>

### 더한 후

- 자리마다 위치 벡터를 하나씩 두고 Token 벡터에 더한다. GPT-2는 위치 벡터도 Embedding처럼 표로 두고 학습으로 값을 정한다.

```text
위치 벡터 표 (칸을 3개로 줄인 예)
  0번 자리  [0.1, 0.0, 0.0]
  2번 자리  [0.0, 0.0, 0.3]

사과 벡터 [0.9, 0.8, 0.1]
  문장 B의 0번 자리 사과 = [0.9, 0.8, 0.1] + [0.1, 0.0, 0.0] = [1.0, 0.8, 0.1]
  문장 A의 2번 자리 사과 = [0.9, 0.8, 0.1] + [0.0, 0.0, 0.3] = [0.9, 0.8, 0.4]
```

- 같은 사과지만 두 문장에서 서로 다른 벡터가 된다.
- 이제 이/가가 바로 앞 Token에 붙는다는 관계를 계산할 수 있고 두 문장이 구분된다.
- 최근 LLM 다수는 위치 벡터를 더하는 대신 벡터를 자리만큼 회전시키는 RoPE 방식을 쓴다. 순서를 알려 준다는 목적은 같다.

<sub>RoPE(Rotary Position Embedding, 회전 위치 임베딩): 자리 번호에 비례하는 각도로 벡터를 돌려서 위치를 표시하는 방식</sub>

### 이 단계에서 바뀐 것

- 순서를 구분할 수 있게 됐다.
- 아직 남은 문제: 각 Token은 자기 정보만 가지고 있다. 세 번째 질문의 사람은 자기가 누구를 가리키는지 모른다.

---

## 위치 정보를 따로 넣는 이유

> Transformer는 앞 내용이 흐려지는 RNN의 한계를 없애려고 문장 전체를 한꺼번에 보는 방식을 택했다. 한꺼번에 보면 순서가 사라지므로 계산에 들어가기 전에 위치 정보를 먼저 더한다.

### Transformer 이전의 RNN

```text
민수 → 는 → 사과 → 를 → 좋아해 → … → 그 → 사람 → 에게 → 뭘 → 사주면
기억 → 기억 → 기억 → …  (단계마다 기억을 고쳐 쓰며 넘김)  … → 기억
```

- RNN은 Token을 앞에서부터 하나씩 처리한다. 지금까지 읽은 내용은 크기가 정해진 벡터 하나에 계속 고쳐 쓰며 넘긴다.
- 읽는 순서 자체가 순서 정보가 되므로 위치를 따로 넣을 필요가 없다.
- 대신 입력이 길어질수록 앞 내용이 흐려진다. 사주면까지 읽었을 때는 첫 문장의 민수와 사과 정보가 많이 약해져 있다.
- 앞 Token을 처리해야 다음 Token을 처리할 수 있어서 동시에 계산하기 어렵다. 그래서 모델을 크게 키우기도 어려웠다.

<sub>RNN(Recurrent Neural Network, 순환 신경망): 입력을 순서대로 하나씩 처리하면서 이전 단계의 기억을 다음 단계로 넘기는 신경망</sub>

### Transformer (2017, 논문 Attention Is All You Need)

- RNN을 빼고 Attention만으로 문장을 처리한다. 각 Token이 다른 모든 Token을 직접 보고 얼마나 참고할지를 비율로 정한다.
- 세 문장 앞의 사과도 바로 앞 Token과 똑같이 한 번에 닿는다. 거리 때문에 정보가 흐려지지 않는다.
- 모든 Token을 동시에 계산할 수 있어서 GPU로 빠르게 학습할 수 있다.
- 다만 한꺼번에 보면 순서가 사라진다. 그래서 3단계에서 위치 정보를 먼저 더해 준다.

| | RNN | 위치 정보 없는 Attention | 위치 정보 + Attention (Transformer) |
|---|---|---|---|
| 먼 앞 내용 | 흐려짐 | 그대로 닿음 | 그대로 닿음 |
| 순서 | 앎 | 모름 | 앎 |
| 동시 계산 | 어려움 | 가능 | 가능 |

<sub>Attention(어텐션): 각 Token이 다른 Token을 얼마나 참고할지 비율로 계산해서 정보를 가져오는 연산. 같은 입력 안에서 하면 Self-Attention이라고 부른다<br>
GPU: 많은 계산을 동시에 처리하는 연산 장치. 원래 그래픽 처리용이었다</sub>

### 이후 변화

- 동시 계산이 가능해지자 모델 크기와 학습 데이터를 크게 늘릴 수 있게 됐다.
- 2020년에는 모델 크기, 데이터 양, 계산량을 늘리면 성능이 예측 가능한 비율로 좋아진다는 연구(Scaling Law)가 나왔다.

```text
2017  Transformer 발표 (번역 모델)
2018  GPT-1      가중치 1.17억 개
2019  GPT-2      15억 개
2020  GPT-3      1,750억 개
2022  ChatGPT 출시
2023~ GPT-4, Claude, Llama, Gemini 등
```

- Token → Embedding → 층 반복 → 다음 Token 예측이라는 기본 구조는 GPT-1부터 지금까지 같다. 바뀐 것은 주로 규모와 세부 부품이다.

---

## 4. Self-Attention

> 각 Token이 차례로 질문하는 쪽(Query)이 되어 앞쪽 Token들의 Key와 비교한다. 관련이 높은 Token의 Value를 더 많이 가져와 자기 벡터에 더한다.

### Query, Key, Value

| 구분 | 뜻 | 예시 (사람 Token이 질문할 때) |
|---|---|---|
| Query | 지금 Token이 찾는 정보 | 내가 가리키는 대상이 누구인가 |
| Key | 각 Token이 가진 정보의 특징. Query와 비교하는 데 쓴다 | 민수의 Key: 사람 이름이다 |
| Value | 선택됐을 때 실제로 넘겨주는 정보 | 민수의 Value: 민수에 관한 정보 |

<sub>Query(쿼리, 질의) / Key(키) / Value(밸류, 값). 줄여서 Q, K, V</sub>

### Q, K, V를 만드는 방법

```text
사람 벡터 × Wq = 사람의 Query
사람 벡터 × Wk = 사람의 Key
사람 벡터 × Wv = 사람의 Value
```

- 한 Token 벡터에 서로 다른 행렬 세 개를 곱해서 만든다. 모든 Token이 같은 Wq, Wk, Wv를 쓴다.
- 행렬 값도 처음엔 무작위이고 학습으로 정해진다.

<sub>행렬: 숫자를 가로세로 표 모양으로 늘어놓은 것. 벡터에 행렬을 곱하면 새 벡터가 나온다. 이렇게 벡터에 행렬을 곱하는 계산을 Linear(선형 변환)라고 부른다</sub>

### 계산 순서

1. 한 Token이 Query가 된다.
2. 그 Query를 자기 자신과 앞쪽 Token들의 Key와 하나씩 내적한다. 이 값이 관련도 점수다.
3. 점수를 Softmax에 넣어 합이 1인 비율로 바꾼다.
4. 비율만큼 각 Token의 Value를 가져와 더한다. 이 결과를 원래 벡터에 더한다.
5. 모든 Token이 Query가 되어 1~4를 한다. 실제로는 행렬 계산으로 모든 Token을 동시에 처리한다.

- 점수는 Softmax에 넣기 전에 벡터 칸 수의 제곱근으로 나눈다. 칸 수가 많으면 내적 값이 커져서 비율이 한 Token에 지나치게 쏠리기 때문이다.

<sub>Softmax(소프트맥스): 점수 목록을 0~1 사이, 합이 1인 비율로 바꾸는 함수. 점수가 높을수록 비율이 커지고 점수 차이가 크면 비율 차이는 더 크게 벌어진다</sub>

### 앞쪽만 보는 규칙 (Causal Mask)

```text
              참고하는 대상 →
질문하는 Token   민수   는   사과   를
민수              O     X     X     X
는                O     O     X     X
사과              O     O     O     X
를                O     O     O     O
```

- 각 Token은 자기 자신과 앞쪽 Token만 본다. X 자리는 점수를 마이너스 무한대로 바꿔서 Softmax 비율이 0이 되게 한다.
- 이유: 학습할 때 모델이 맞혀야 하는 정답이 바로 다음 Token이다. 뒤쪽을 볼 수 있으면 정답을 보고 맞히는 셈이 되어 학습이 되지 않는다.
- 이렇게 앞쪽만 보도록 제한한 Attention을 Causal Attention이라고 한다. GPT 같은 생성 모델은 모두 이 방식을 쓴다.

<sub>Causal Mask(인과 마스크): 뒤쪽 Token을 가리는 표. Causal Attention(인과적 어텐션): 이 표를 적용해 앞쪽만 참고하는 Attention. 인과(causal)는 앞의 것이 뒤의 것에만 영향을 준다는 뜻</sub>

### 예시 1. 사람이 Query일 때

```text
Query(사람): 내가 가리키는 대상이 누구인가

비교한 Key     점수(Q·K)   Softmax 비율
민수            5.0          0.74
그              3.2          0.12
과일 가게       2.8          0.08
사과            2.4          0.06

가져오는 정보 = 0.74×V(민수) + 0.12×V(그) + 0.08×V(과일 가게) + 0.06×V(사과)
새 사람 벡터 = 원래 사람 벡터 + 가져오는 정보
```

- 적용 전: 사람은 누구인지 정해지지 않은 일반 명사다.
- 적용 후: 사람 벡터에 민수 정보가 74% 비율로 섞였다. 그 사람 = 민수가 연결된다.

### 예시 2. 사주면이 Query일 때

```text
Query(사주면): 무엇을 사 줘야 하는가

비교한 Key               Softmax 비율
사과                     0.55
좋아해                   0.20
사람 (민수 정보가 섞임)  0.15
그 밖의 Token            0.10
```

- 적용 전: 사주면은 무언가를 사 준다는 일반적인 뜻이다.
- 적용 후: 사주면 벡터에 사과와 좋아함 정보가 섞였다.

### 2단계에서 남은 문맥 문제 해결

- 배가 고파서 사과를: 사과가 Query가 되어 앞쪽 배가 고파서의 Value를 가져온다. 사과 벡터가 과일 쪽 의미가 된다.
- 약속에 늦어서 사과를: 같은 방식으로 약속에 늦어서의 Value를 가져온다. 사과 벡터가 사죄 쪽 의미가 된다.
- 같은 31번 Token이 앞 문맥에 따라 다른 벡터가 된다.

### 거리와 상관없이 참고한다

- 비율은 거리가 아니라 Query와 Key가 얼마나 맞는지로 정해진다.
- 세 문장 앞의 민수와 사과도 바로 옆 Token보다 높은 비율을 받을 수 있다.

### 이 단계에서 바뀐 것

- Token끼리 정보를 주고받게 됐다. 그 사람이 누구인지, 무엇을 사 줘야 하는지가 벡터에 들어갔다.
- 아직 남은 문제: 비율을 한 세트만 정할 수 있어서 여러 관계를 한꺼번에 강하게 챙기기 어렵다.

---

## 5. Multi-Head Attention

> Self-Attention을 여러 갈래(Head)로 나눠 동시에 한다. 갈래마다 다른 관계를 본다.

### Head가 하나일 때

```text
사주면이 참고하는 비율 (한 세트)
  민수 0.45   사과 0.40   그 밖 0.15
```

- 사주면은 누구에게(민수)와 무엇을(사과)을 둘 다 알아야 한다.
- 비율이 한 세트뿐이라 둘이 나눠 갖는다. 어느 쪽도 뚜렷하게 들어오지 않는다.

### Head가 여러 개일 때

```text
          ┌ Head 1: 누구에게 주는가     민수 0.85
사주면 ───┼ Head 2: 무엇을 주는가       사과 0.80
          ├ Head 3: 문장 구조 (주어, 목적어)
          └ …
                    ↓ 결과를 이어 붙인 뒤 Linear 한 번
              사주면의 새 벡터
```

- 벡터를 Head 수만큼 조각내서 각 조각이 따로 Q, K, V와 비율을 계산한다. GPT-2는 768칸을 12개 Head가 64칸씩 나눠 쓴다.
- 갈래마다 비율을 따로 정하므로 민수와 사과를 각각 뚜렷하게 가져온다.
- 어떤 Head가 어떤 관계를 볼지는 사람이 정하지 않는다. 학습 중에 나뉜다. 위의 Head별 역할은 이해를 위한 예시다.

<sub>Head(헤드): Q, K, V 계산 한 세트. Multi-Head는 이 세트를 여러 개 두는 것</sub>

### 이 단계에서 바뀐 것

- 한 Token이 여러 종류의 관계를 동시에 가져온다.
- 아직 남은 문제: 가져온 정보는 여러 Token의 Value가 섞인 상태다. 다음 계산에 쓰기 좋게 정리할 필요가 있다.

---

## 6. FFN

> Attention이 다른 Token에서 정보를 가져오는 단계라면 FFN은 각 Token이 가져온 정보를 따로 가공하는 단계다.

| 구분 | Attention | FFN |
|---|---|---|
| 다른 Token을 보나 | 본다 | 보지 않는다 |
| 하는 일 | 정보를 가져온다 | 가져온 정보를 가공한다 |
| 계산 | Q·K 비교, Value 가중합 | Linear → 활성화 함수 → Linear |

### 계산 과정

```text
사주면 벡터 (768칸)
  ↓ Linear: 768칸 → 3072칸으로 넓힘
  ↓ GELU: 음수는 거의 0으로 줄이고 양수는 대부분 통과
       예) [-2.0, 0.5, 3.0] → [-0.05, 0.35, 3.00]
  ↓ Linear: 3072칸 → 768칸으로 되돌림
가공된 사주면 벡터 (768칸)
```

- 칸을 넓히면 섞여 있는 정보를 더 많은 조합으로 나눠 따져 볼 수 있다. GELU가 그중 약한 조합을 거의 0으로 줄인다. 마지막 Linear가 남은 결과를 원래 크기로 모은다.
- 예: Attention 뒤의 사주면 벡터에는 민수, 사과, 좋아함, 과일 가게 정보가 섞여 있다. FFN은 이것을 다음 층에서 쓰기 좋은 형태로 바꾼다.
- GPT-2 기준으로 층 하나의 가중치 중 약 2/3가 FFN에 있다. 학습한 사실과 패턴이 많이 저장되는 곳으로 분석된다.

<sub>활성화 함수: Linear 사이에 넣는 함수. 이것이 없으면 Linear를 여러 번 해도 Linear 한 번과 결과가 같아서 복잡한 관계를 표현하지 못한다<br>
GELU(Gaussian Error Linear Unit): GPT 계열이 쓰는 활성화 함수. 최근 모델은 비슷한 역할의 SwiGLU를 많이 쓴다</sub>

### 이 단계에서 바뀐 것

- 가져온 정보가 다음 계산에 쓰기 좋은 형태로 정리됐다.
- 아직 남은 문제: 이 계산을 수십 층 쌓으면 처음 정보가 사라지고 값이 너무 커지거나 작아질 수 있다.

---

## 7. Residual Connection, LayerNorm

> 원래 벡터에 새로 계산한 값을 더하고 값의 크기를 맞춘다. 층을 깊게 쌓아도 처음 정보가 남고 계산이 안정된다.

### Residual Connection

```text
원래 사람 벡터              [0.2, 0.5, 0.1]
Attention이 가져온 정보     [0.6, 0.1, 0.0]
더한 결과                   [0.8, 0.6, 0.1]
```

- 더하지 않고 새 값으로 바꿔 버리면 층을 지날 때마다 원래 Token 정보가 덮어써진다. 수십 층 뒤에는 이 자리가 원래 무슨 Token이었는지도 흐려진다.
- 더하면 원래 정보는 그대로 남고 새 정보만 쌓인다.
- 학습할 때도 고칠 방향이 이 덧셈 경로를 타고 앞쪽 층까지 잘 전달된다. 그래서 수십~100여 층을 쌓아도 학습이 된다.

### LayerNorm

```text
정규화 전  [10.0,  0.1, -8.0]
정규화 후  [ 1.26, -0.08, -1.18]   평균 0, 크기가 고르게 맞춰짐
```

- 층을 거치며 값이 지나치게 커지거나 작아지면 계산이 불안정해진다.
- 각 벡터의 평균을 0, 퍼진 정도를 1로 맞춰서 다음 층이 늘 비슷한 크기의 값을 받게 한다.
- 최근 모델은 계산이 더 단순한 RMSNorm을 많이 쓴다.

<sub>정규화: 값의 범위를 일정하게 맞추는 계산<br>
RMSNorm(Root Mean Square Normalization): 평균은 맞추지 않고 크기만 맞추는 정규화</sub>

### 층(Layer) 하나의 구성

```text
입력 벡터 x
  → LayerNorm → Attention(Multi-Head) → x에 더함
  → LayerNorm → FFN                  → x에 더함
출력 벡터 x
```

---

## 8. Layer 반복

> 같은 구조의 층을 수십 번 반복한다. 층마다 Value가 전달되고 쌓이면서 여러 번 건너야 이어지는 관계도 연결된다.

### 층이 하나뿐일 때

- 직접 연결된 정보를 한 번만 가져온다.
- 마지막 자리가 사람을 참고해도 그 시점의 사람 벡터에는 아직 민수 정보가 없을 수 있다.

### 여러 층일 때

```text
1층   사람        ← 민수 정보를 가져옴                    그 사람 = 민수
2층   사주면      ← 민수 정보가 섞인 사람, 사과를 가져옴   민수에게 사과
3층~  마지막 자리 ← 위 결과가 쌓인 Token들을 가져옴
…
N층   마지막 자리의 최종 벡터
```

- 정보가 사과 → 민수 → 그 사람 → 마지막 자리로 여러 단계를 건너 전달된다.
- 층마다 역할이 이렇게 깔끔하게 나뉘지는 않는다. 실제로는 여러 Head와 층에 흩어져 있다.
- Token 자체가 다른 Token으로 바뀌지는 않는다. 같은 자리의 벡터가 층마다 새로 고쳐진다.
- GPT-2는 12층, 현재 대형 LLM은 수십~100여 층을 쌓는다.

<sub>Hidden State(은닉 상태): 층을 거치며 고쳐지는 각 자리의 벡터. 보통 마지막 층의 출력을 가리킨다</sub>

---

## 9. 출력

> 마지막 자리의 벡터로 사전 속 모든 Token의 점수를 계산한다. 점수를 확률로 바꾼 뒤 하나를 고른다.

- 마지막 자리만 쓰는 이유: 앞쪽만 보는 규칙 때문에 마지막 자리가 앞의 모든 Token 정보를 받은 유일한 자리다.

```text
마지막 자리 벡터 (768칸)
  ↓ LayerNorm → lm_head (768칸 → 사전 크기 50,257칸)
Logits: 사전의 모든 Token마다 점수 하나
  사과    8.4
  바나나  5.1
  포도    4.0
  꽃      3.2
  …
  ↓ Softmax (위 4개만으로 계산한 예)
확률: 사과 95%, 바나나 3.5%, 포도 1.2%, 꽃 0.5%
  ↓ 고르기
사과
```

<sub>Logits(로짓): Softmax에 넣기 전의 원점수<br>
lm_head(출력층): 마지막 벡터를 사전 크기만큼의 점수로 바꾸는 Linear. GPT-2는 Embedding 표와 같은 숫자를 함께 쓴다</sub>

### 고르는 방법

- 가장 높은 것 고르기: 늘 사과를 고른다. 같은 질문에 늘 같은 답이 나온다.
- 확률대로 뽑기: 95% 확률로 사과, 3.5% 확률로 바나나가 나온다. 같은 질문에도 답이 달라질 수 있는 이유다.
- Top-k: 점수가 높은 k개 후보 안에서만 뽑는다. 이 저장소 코드는 상위 50개 안에서 뽑는다.
- Temperature: 점수를 이 값으로 나눈 뒤 Softmax에 넣는다. 값이 클수록 확률이 고르게 퍼져서 다양한 답이 나온다.

```text
Temperature 1.0   사과 95%    바나나 3.5%    포도 1.2%   꽃 0.5%
Temperature 2.0   사과 73%    바나나 14%     포도 8%     꽃 5%
```

<sub>Top-k(상위 k개): 후보를 점수 순으로 k개만 남기고 나머지는 버리는 방식<br>
Temperature(온도): 확률 분포를 뾰족하게 또는 평평하게 조절하는 값. 1보다 작으면 한 후보에 더 쏠리고 크면 고르게 퍼진다</sub>

---

## 10. 생성 반복

> LLM은 Token을 한 번에 하나만 만든다. 고른 Token을 입력 뒤에 붙여 다음 Token을 다시 계산한다.

```text
그럼 그 사람에게 뭘 사주면 좋을까?
                    ↓
                  사과
```

이제 사과가 들어간 문맥으로 다음 Token을 계산한다.

```text
… 뭘 사주면 좋을까? 사과
                    ↓
                    를
                    ↓
                 추천해요
```

```text
문맥 → 1~9단계 계산 → 다음 Token 하나 → 문맥 뒤에 붙임 → 다시 계산 → … → 끝 표시 Token
```

- 끝 표시 Token이 나오거나 정해 둔 최대 길이에 닿으면 멈춘다. Llama 3에서는 1장의 <|eot_id|>가 끝 표시다.
- 실제 LLM은 매번 처음부터 다시 계산하지 않는다. 앞 Token들의 Key, Value를 KV Cache에 저장해 두고 새로 붙은 Token만 계산한다.
- 한 번에 넣을 수 있는 Token 수에는 한도(Context Window)가 있다. GPT-2는 1,024개, 최근 LLM은 12만 8천 개 이상이다.

<sub>KV Cache(키-값 캐시): 앞 Token들의 Key, Value를 저장해 두는 공간. 새 Token은 저장된 Key, Value와만 비교하면 된다<br>
Context Window(컨텍스트 윈도우): 모델이 한 번에 받아서 참고할 수 있는 최대 Token 수</sub>

---

## 11. 학습

> Embedding 표, Wq·Wk·Wv, FFN 등 모든 가중치는 무작위 값에서 시작한다. 다음 Token을 맞히도록 조금씩 고치는 일을 아주 많이 반복해서 값이 정해진다.

### 사전학습

```text
입력: 민수는 사과를        정답: 좋아해
  ↓ 1~9단계 계산
모델이 준 좋아해의 확률     학습 초기 2%   →  학습 후 60%
Loss (= -log 확률)          3.9            →  0.51
  ↓ 역전파
Loss가 줄어드는 방향으로 모든 가중치를 조금씩 고침
  ↓
수조 개 Token 분량의 글로 반복
```

- 다음 Token을 잘 맞히려다 보니 문법, 그 사람 같은 말이 가리키는 대상, 사실 관계 같은 패턴이 가중치에 담긴다.

<sub>사전학습(Pre-training): 많은 양의 글로 다음 Token 맞히기를 학습하는 첫 단계<br>
Loss(손실): 예측이 정답과 얼마나 다른지 나타내는 값. 정답 확률이 높을수록 작아진다<br>
역전파(Backpropagation): Loss를 줄이려면 각 가중치를 어느 방향으로 얼마나 고쳐야 하는지 출력 쪽에서 입력 쪽으로 거슬러 계산하는 방법</sub>

### 대화형으로 만드는 추가 학습

- 사전학습만 한 모델은 글을 이어 쓰기만 한다. 질문을 넣으면 답 대신 비슷한 질문을 이어 쓰기도 한다.
- 질문과 좋은 답변 예시로 추가 학습한다(SFT). 이때 1장의 화자 구분 특수 Token 형식도 익힌다.
- 사람이 더 낫다고 고른 답변 쪽으로 한 번 더 학습한다(RLHF).
- ChatGPT, Claude 같은 서비스는 이 과정을 모두 거친 모델이다.

| 구분 | 학습 (Training) | 사용 (Inference) |
|---|---|---|
| 목적 | 가중치 값을 정한다 | 정해진 가중치로 답을 만든다 |
| 흐름 | 예측 → Loss → 역전파 → 수정 | 예측 → 고르기 → 붙이기 → 반복 |

<sub>SFT(Supervised Fine-Tuning, 지도 미세조정): 정답 예시가 있는 데이터로 추가 학습하는 것<br>
RLHF(Reinforcement Learning from Human Feedback, 인간 피드백 기반 강화학습): 사람이 매긴 선호를 기준으로 더 나은 답변을 내도록 학습하는 것<br>
Inference(추론): 학습이 끝난 모델로 결과를 만드는 것</sub>

---

## 12. 단계별 데이터 변화

T: Token 수, 768: GPT-2의 벡터 칸 수, 50,257: GPT-2의 사전 크기

| 단계 | 데이터 모양 | 예시에서 일어난 일 |
|---|---|---|
| 입력 | 글자 | 대화 전체 |
| 1. Tokenizer | 정수 T개 | 민수는 사과를 좋아해. → [12, 7, 31, 8, 45, 2] |
| 2. Embedding | T × 768 | 사과 → [0.9, 0.8, 0.1] |
| 3. 위치 정보 | T × 768 | 자리마다 다른 벡터가 됨 |
| 4~5. Attention | T × 768 | 사람에 민수 정보, 사주면에 사과 정보가 들어감 |
| 6. FFN | T × 768 | 섞인 정보가 정리됨 |
| 7. Residual, LayerNorm | T × 768 | 원래 정보 유지, 값 크기 정리 |
| 8. 층 반복 (12층) | T × 768 | 사과 → 민수 → 그 사람 → 마지막 자리 |
| 9. 출력 | 50,257개 점수 → 확률 | 사과 95% |
| 10. 생성 반복 | Token이 하나씩 늘어남 | 사과 → 를 → 추천해요 |

- Transformer는 각 Token이 Query로 필요한 정보를 찾고 Key로 대상을 고른 뒤 Value를 가져와 자기 벡터에 쌓는 일을 여러 층 반복하는 구조다.
- LLM은 그렇게 만든 마지막 자리의 벡터로 다음 Token을 하나씩 고르는 일을 반복한다.

---

## 13. 코드 대응 (train_gpt2.py)

| 단계 | 줄 | 코드 |
|---|---|---|
| 2. Embedding, 3. 위치 벡터 표 | 86–87 | `wte = nn.Embedding(vocab_size, n_embd)`, `wpe = nn.Embedding(block_size, n_embd)` |
| 처음 값을 무작위로 채움 | 99–108 | `torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)` |
| Token + 위치 | 116–118 | `x = tok_emb + pos_emb` |
| 4. Q, K, V 만들기 | 31–32 | `qkv = self.c_attn(x)`, `q, k, v = qkv.split(...)` |
| 5. Head 나누기 | 33–35 | `.view(B, T, n_head, C // n_head)` |
| 4. 앞쪽만 보는 Attention | 36 | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| 6. FFN | 46–48 | `Linear(768→3072)` → `GELU` → `Linear(3072→768)` |
| 7. Residual + LayerNorm | 67–68 | `x = x + self.attn(self.ln_1(x))`, `x = x + self.mlp(self.ln_2(x))` |
| 8. 층 반복 | 120–121 | `for block in self.transformer.h: x = block(x)` |
| 9. Logits | 123–124 | `ln_f` → `lm_head` |
| 9. 출력층과 Embedding 공유 | 94 | `wte.weight = lm_head.weight` |
| 9~10. 고르기, 생성 반복 | 461–473 | 마지막 자리 logits → softmax → 상위 50개 중 뽑기 → `torch.cat`으로 붙이기 |
| 11. Loss | 125–127 | `F.cross_entropy(logits, targets)` |
| 설정값 | 72–77 | `vocab_size=50257, n_layer=12, n_head=12, n_embd=768` |

---

## 14. 직접 해보기

모두 GitHub에 공개된 오픈소스다. 위에서부터 쉬운 순서다.

| 순서 | 해볼 것 | 관련 단계 | 필요한 것 | 걸리는 시간 |
|---|---|---|---|---|
| 1 | Tiktokenizer로 Token 나뉘는 모습 보기 | 1 | 브라우저 | 10분 |
| 2 | Transformer Explainer로 전체 흐름 보기 | 1~10 | 브라우저 | 30분 |
| 3 | tiktoken으로 한국어, 영어 Token 수 비교 | 1 | Python | 15분 |
| 4 | minbpe로 BPE Tokenizer 직접 학습 | 1 | Python | 30분 |
| 5 | BertViz로 GPT-2의 Attention 비율 보기 | 4~5 | Python, Jupyter | 30분 |
| 6 | nanoGPT로 작은 GPT 학습하고 문장 생성 | 2~11 | Python, PyTorch (CPU 가능) | 1시간 |
| 7 | 이 저장소의 커밋을 순서대로 따라가기 | 전체 | Git | 자유 |

### 1. Tiktokenizer

- 주소: https://tiktokenizer.vercel.app (GitHub: dqbd/tiktokenizer)
- 해볼 것: 모델을 gpt2와 gpt-4o로 바꿔 가며 민수는 사과를 좋아해.를 입력한다. Token 개수와 나뉘는 위치를 비교한다.

### 2. Transformer Explainer

- 주소: https://poloclub.github.io/transformer-explainer (GitHub: poloclub/transformer-explainer)
- 브라우저 안에서 실제 GPT-2가 돌아간다. 입력한 문장이 Embedding, Q·K·V, 앞쪽만 보는 마스크, Softmax, FFN, 확률로 바뀌는 과정을 화면으로 보여 준다.
- 해볼 것
  - 영어 문장을 넣고 Attention 부분을 펼쳐 각 Token이 앞쪽 Token에만 비율을 주는지 확인한다(앞쪽만 보는 규칙).
  - Temperature 슬라이더를 움직여 다음 Token 확률이 퍼지고 모이는 모습을 본다.
- 3D로 보고 싶다면 LLM Visualization(https://bbycroft.net/llm, GitHub: bbycroft/llm-viz)도 있다.

### 3. tiktoken으로 Token 수 비교

```bash
pip install tiktoken
```

```python
import tiktoken

for name in ["gpt2", "o200k_base"]:          # o200k_base = GPT-4o Tokenizer
    enc = tiktoken.get_encoding(name)
    for s in ["Minsu likes apples.", "민수는 사과를 좋아해."]:
        ids = enc.encode(s)
        print(name, s, len(ids), ids)
        print("  ", [enc.decode([i]) for i in ids])
```

- 예상 결과: gpt2는 영어 6개, 한국어 26개 / o200k_base는 영어 5개, 한국어 9개

### 4. minbpe로 BPE 직접 학습

```bash
git clone https://github.com/karpathy/minbpe.git
cd minbpe
```

```python
from minbpe import BasicTokenizer

text = open("korean.txt", encoding="utf-8").read()   # 한국어 글 아무거나 몇 KB
tok = BasicTokenizer()
tok.train(text, 256 + 100)          # 바이트 256개에서 시작해 100번 합치기
print(len("민수는 사과를 좋아해.".encode("utf-8")))   # 합치기 전: 바이트 수 30
print(len(tok.encode("민수는 사과를 좋아해.")))        # 합친 뒤 Token 수
tok.save("ko")                      # ko.vocab 파일에서 어떤 조합이 합쳐졌는지 확인
```

- 합치는 횟수를 늘리면 Token 수가 어떻게 줄어드는지 본다.

### 5. BertViz로 Attention 비율 보기

```bash
pip install bertviz transformers torch jupyterlab ipywidgets
```

```python
from transformers import AutoTokenizer, AutoModel
from bertviz import head_view

model = AutoModel.from_pretrained("gpt2", output_attentions=True,
                                  attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
inputs = tokenizer.encode("The animal didn't cross the street because it was too tired",
                          return_tensors="pt")
attention = model(inputs).attentions
tokens = tokenizer.convert_ids_to_tokens(inputs[0])
head_view(attention, tokens)        # Jupyter에서 실행
```

- 해볼 것: it을 눌러 층과 Head를 바꿔 가며 animal에 높은 비율이 걸리는 곳을 찾는다. 4단계 예시 1(사람 → 민수)과 같은 현상이다.
- GPT-2 Tokenizer는 한국어를 바이트 조각으로 나누므로 이 실습은 영어 문장이 보기 편하다.

### 6. nanoGPT로 작은 GPT 학습하고 생성

```bash
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
pip install torch numpy transformers datasets tiktoken wandb tqdm

python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
  --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 \
  --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

- 셰익스피어 글을 글자 단위 Token으로 학습한다. CPU에서 3분 정도 걸린다(nanoGPT 안내 기준).
- 해볼 것
  - 학습 중 출력되는 loss가 줄어드는 것을 본다(11단계).
  - --max_iters=200처럼 짧게 학습한 모델과 생성 결과를 비교한다.
  - sample.py에 --temperature=0.5, --temperature=1.5를 붙여 생성 결과가 어떻게 달라지는지 본다(9단계).
  - --n_layer, --n_head를 바꿔 층과 Head 수가 결과에 주는 영향을 본다.
- 공개된 GPT-2로 바로 생성해 볼 수도 있다.

```bash
python sample.py --init_from=gpt2 --start="Minsu likes apples. He went to" \
  --num_samples=3 --max_new_tokens=30 --device=cpu
```

### 7. 이 저장소의 커밋 따라가기

```bash
git log --reverse --oneline master
```

- 이 저장소는 빈 파일에서 GPT-2 재현까지 한 단계씩 커밋해 두었다. 커밋을 순서대로 checkout하며 train_gpt2.py가 늘어나는 과정을 볼 수 있다.
- 같은 내용의 강의 영상이 README에 연결되어 있다(https://youtu.be/l8pRSuU81PU).
- 전체 학습(FineWeb 데이터 100억 Token)은 GPU가 필요하다. 코드를 읽고 13장의 줄 번호와 맞춰 보는 용도로 쓰기 좋다.

---

## References

1. Vaswani et al., Attention Is All You Need, 2017. https://arxiv.org/abs/1706.03762
2. Radford et al., Language Models are Unsupervised Multitask Learners (GPT-2), 2019
3. Brown et al., Language Models are Few-Shot Learners (GPT-3), 2020. https://arxiv.org/abs/2005.14165
4. Kaplan et al., Scaling Laws for Neural Language Models, 2020. https://arxiv.org/abs/2001.08361
5. Ouyang et al., Training language models to follow instructions with human feedback, 2022. https://arxiv.org/abs/2203.02155
6. Llama Team, The Llama 3 Herd of Models, 2024. https://arxiv.org/abs/2407.21783
7. Andrej Karpathy, build-nanogpt / nanoGPT / minbpe. https://github.com/karpathy
8. Cho et al., Transformer Explainer, 2024. https://arxiv.org/abs/2408.04619
