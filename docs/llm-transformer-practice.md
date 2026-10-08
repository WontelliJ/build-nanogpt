# LLM 동작 원리: 코드 대응과 실습

- 본문(llm-transformer-wiki.md)의 각 단계가 실제 코드 어디에 있는지, 직접 해볼 수 있는 오픈소스 실습, 본문에서 뺀 심화 내용을 모았다.

---

## 코드 대응 (train_gpt2.py)

| 단계 | 줄 | 코드 |
|---|---|---|
| Embedding 표, 위치 벡터 표 | 86–87 | `wte = nn.Embedding(vocab_size, n_embd)`, `wpe = nn.Embedding(block_size, n_embd)` |
| 처음 값을 무작위로 채움 | 99–108 | `torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)` |
| Token 벡터 + 위치 벡터 | 116–118 | `x = tok_emb + pos_emb` |
| Self-Attention: Q, K, V 만들기 | 31–32 | `qkv = self.c_attn(x)`, `q, k, v = qkv.split(...)` |
| Multi-Head: Head 나누기 | 33–35 | `.view(B, T, n_head, C // n_head)` |
| Self-Attention: 앞쪽만 보기 | 36 | `F.scaled_dot_product_attention(q, k, v, is_causal=True)` |
| Feed Forward Network | 46–48 | `Linear(768→3072)` → `GELU` → `Linear(3072→768)` |
| Layer Normalization 후 계산, Residual로 더하기 | 67–68 | `x = x + self.attn(self.ln_1(x))`, `x = x + self.mlp(self.ln_2(x))` |
| Transformer Block × N | 120–121 | `for block in self.transformer.h: x = block(x)` |
| Logits | 123–124 | `ln_f` → `lm_head` |
| 출력층과 Embedding 표 공유 | 94 | `wte.weight = lm_head.weight` |
| Probability, Next Token, 반복 | 461–473 | 마지막 자리 logits → softmax → 상위 50개 중 뽑기 → `torch.cat`으로 붙이기 |
| 학습: Loss | 125–127 | `F.cross_entropy(logits, targets)` |
| 설정값 | 72–77 | `vocab_size=50257, n_layer=12, n_head=12, n_embd=768` |

---

## 직접 해보기

모두 GitHub에 공개된 오픈소스다. 위에서부터 쉬운 순서다.

| 순서 | 해볼 것 | 관련 단계 | 필요한 것 | 걸리는 시간 |
|---|---|---|---|---|
| 1 | Tiktokenizer로 Token 나뉘는 모습 보기 | Token | 브라우저 | 10분 |
| 2 | Transformer Explainer로 전체 흐름 보기 | 전체 | 브라우저 | 30분 |
| 3 | tiktoken으로 한국어, 영어 Token 수 비교 | Token | Python | 15분 |
| 4 | minbpe로 BPE Tokenizer 직접 학습 | Token | Python | 30분 |
| 5 | BertViz로 GPT-2의 Attention 비율 보기 | Self-Attention, Multi-Head | Python, Jupyter | 30분 |
| 6 | nanoGPT로 작은 GPT 학습하고 문장 생성 | 전체, 학습 | Python, PyTorch (CPU 가능) | 1시간 |
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

- 해볼 것: it을 눌러 층과 Head를 바꿔 가며 animal에 높은 비율이 걸리는 곳을 찾는다. 본문 Self-Attention 예시(사람 → 민수)와 같은 현상이다.
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
  - 학습 중 출력되는 loss가 줄어드는 것을 본다(학습).
  - --max_iters=200처럼 짧게 학습한 모델과 생성 결과를 비교한다.
  - sample.py에 --temperature=0.5, --temperature=1.5를 붙여 생성 결과가 어떻게 달라지는지 본다(Probability).
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
- 전체 학습(FineWeb 데이터 100억 Token)은 GPU가 필요하다. 코드를 읽고 위 코드 대응 표의 줄 번호와 맞춰 보는 용도로 쓰기 좋다.

---

## 본문에서 뺀 심화 내용

본문을 짧게 유지하려고 뺀 내용이다. 단계 순서대로 적었다.

### Token: 바이트 단위 BPE

- 사전의 출발점은 정확히는 글자가 아니라 바이트 256개다. 그래서 사전에 없는 글자도 바이트 조각을 이어 붙여 만들 수 있다.
- 문장을 자를 때는 사전을 만들 때 배운 합치기 순서를 그대로 적용한다.

### Position Information: 왜 더하나, RNN, RoPE

- 위치 벡터를 이어 붙이지 않고 더하면 벡터 칸 수가 늘지 않아 이후 계산 구조를 그대로 쓸 수 있다. 칸이 수백 개라 학습 과정에서 Token 정보와 위치 정보가 서로 다른 방향을 쓰게 되어 더해도 섞여 사라지지 않는다.
- Transformer 이전에 쓰던 RNN(Recurrent Neural Network, 순환 신경망)은 한 단어씩 순서대로 읽어서 순서는 알았다. 하지만 읽은 내용을 작은 기억 하나에 계속 덮어쓰며 넘겨서 문장이 길면 앞 내용이 흐려졌다.
- 최근 LLM 다수는 RoPE(Rotary Position Embedding, 회전 위치 임베딩)를 쓴다. 내용 벡터에는 위치를 더하지 않고 Attention에서 Query와 Key를 자리 번호에 비례하는 각도로 회전시켜 위치를 반영한다.

### Self-Attention: 용어

- Attention Score: Query와 Key를 비교한 관련도 점수
- Attention Weight: 점수를 Softmax로 바꾼 비율
- Block마다 Query, Key, Value를 만드는 가중치가 따로 있어서 Block마다 찾는 정보가 달라진다.
- 문법 관계 예: "민수는 포도를 싫어하고 사과를 좋아해"에서 '좋아해'는 더 가까운 '포도'보다 바로 앞 목적어 '사과'에 높은 비율을 준다. '포도'는 '싫어하고'가 가져간다.

### Multi-Head Attention: 크기

- GPT-2의 각 Head는 64칸짜리 Query, Key, Value를 만든다. 12개 Head의 결과(64칸 × 12 = 768칸)를 이어 붙인 뒤 가중치를 한 번 더 곱해 하나의 벡터로 합친다.

### Feed Forward Network: GELU

- GPT 계열의 활성화 함수는 GELU다. 음수 점수를 0 가까이 줄인다. 예: 2.1 → 2.06, -0.8 → -0.17, -1.5 → -0.10
- GELU가 줄이는 것은 Token이나 Value가 아니라 3072개 패턴 중 지금 입력과 맞지 않는 패턴이다.

### Residual Connection, Layer Normalization

- Residual Connection은 학습할 때 고쳐야 할 방향(기울기)이 앞쪽 Block까지 잘 전달되게 해 깊은 모델을 학습할 수 있게 한다.
- 최근 모델은 평균은 두고 크기만 맞추는 RMSNorm을 많이 쓴다.

### Next Token: Top-k

- 상위 k개 후보 안에서만 확률대로 뽑는 방식이다. 엉뚱한 Token이 뽑히는 것을 막는다. train_gpt2.py는 상위 50개를 쓴다.

### 반복: Prefill, Decode, KV Cache

| 계산 | 처리하는 Token | 이름 |
|---|---|---|
| 첫 번째 | 질문을 포함한 대화 전체를 한꺼번에 | Prefill |
| 두 번째 이후 | 새로 붙은 Token 하나씩 | Decode |

- Decode 단계에서는 새 Token만 Embedding과 다음 자리 위치 벡터부터 계산한다. 고른 Token은 이미 Token ID라서 자르는 과정이 필요 없다.
- 앞 Token들의 Key, Value는 저장해 두고 Attention에서 다시 쓴다. 이 저장 공간을 KV Cache라고 한다.
- train_gpt2.py처럼 KV Cache 없이 매번 전체 문장을 처음부터 다시 계산하는 단순한 구현도 있다. 결과는 같고 속도만 느리다.
