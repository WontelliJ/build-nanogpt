# LLM 동작 원리: 코드 대응과 실습

- 본문(llm-transformer-wiki.md)의 각 단계가 실제 코드 어디에 있는지, 직접 해볼 수 있는 오픈소스 실습을 모았다.

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
