# 研究に役立つアプリ開発！
AIは論文を要約や検索したり、情報を纏めたり、メールを書いたり...研究でもすごく役に立ちますよね！  
実はLLMを利用すれば研究生活のQOLを更に改善するプログラム開発にも利用できるんです！  
今回はその**アイデア出しから実装まで**ChatGPTを利用して作ってみたいと思います。  
なお、今回はシークレットモードの一時チャットを利用しているので情報の漏洩はありませんのでご安心ください。  

＊今回は分かりやすくするために日本語でプロンプトします。

## 1.アイデア出し
まず、研究生活のQOLを上げるプログラムのアイデアが私には**全くありません**！！  
そこでアイデアをChatGPTことチャッピー君にお願いしましょう。  


プロンプト：  
`あなたは研究者です。研究生活のQOLを更に改善するプログラム開発を行いたいと考えています。pythonプログラムのアイデアを３つ出してください。`  

出力  
<details><summary>折りたたみ1</summary>

研究生活のQOL（Quality of Life）向上を目的としたPythonプログラムのアイデアを3つ提案します。それぞれ、日々の研究活動における負担軽減や効率化を狙ったものです。

---

### 1. 📅 **論文リーディング・トラッカー + 要約ノート自動化ツール**

**概要**:
読んだ論文を記録し、BibTeX管理、メモ、自動要約、タグ付けまでを一元管理するツール。論文のPDFを読み込むと、要約（GPTなどを活用）とキーワード抽出を行い、研究ノートとして記録。

**機能**:

* PDFからタイトル・著者・アブストラクトを自動抽出
* 要約生成（例: ChatGPT API連携）
* タグや研究分野でフィルタリング
* BibTeX情報の自動生成と管理
* ローカルまたはNotion/Obsidian連携

**QOL向上ポイント**:

* 読んだ論文の再検索不要
* 自動要約で時間節約
* 知識の蓄積と検索が効率化

---

### 2. ⏰ **ポモドーロ + 実験ログ記録ツール**

**概要**:
作業時間を管理しつつ、実験ログをタイムスタンプ付きで記録するツール。ポモドーロ方式（25分集中 + 5分休憩）に合わせてメモを促す。

**機能**:

* 作業/休憩タイマーのUI付き
* 各セッション終了時に「何をしたか」を入力 → 自動ログ
* 実験ノート形式（Markdown）で保存
* 毎日の記録を日報/週報としてまとめ出力

**QOL向上ポイント**:

* 実験記録の自動化・可視化
* 作業の見える化でモチベーション維持
* サボり防止にも◎

---

### 3. 🔍 **論文・コードの類似性探索ツール**

**概要**:
自分の研究テーマやキーワードから、arXivやGitHubを自動検索し、似た論文やコードをリストアップするツール。比較的ニッチなトピックにも対応。

**機能**:

* キーワードからarXiv API/GitHub APIを活用し関連情報を収集
* テキストベースの類似度スコア算出（例: SentenceTransformer利用）
* 研究トピックの可視化（クラスターマップ表示）
* 「前に見たことある論文かどうか」の識別

**QOL向上ポイント**:

* 研究アイデアの独自性チェック
* 関連研究のリサーチを効率化
* 実装例の発見が早くなる

---

気になるものや、「これをベースに詳細設計したい」という希望があれば、さらに深掘りしてプロトタイプを一緒に作成することも可能です！

</details>

---
おお、これはすごい。  
自分にはなかったアイデアです。  
あったとしても言語化が難しいものを言語化してくれるので、自分のアイデアを理解するのにもいいですね。  

今回は３つ目の「論文・コードの類似性探索ツール」に近しいものを作ってみようかな。   

## 2.概要の確認
チャッピー君が現状でどんなツールをイメージしているのかわからないのでそれを明確にしましょう。  
プロンプト：  
`3. 🔍 論文・コードの類似性探索ツールをベースに詳細設計を行いたいです。概要について教えてください。`

<details><summary>折りたたみ２</summary>
了解しました！
それでは、**「論文・コードの類似性探索ツール」**の**概要**について、目的・利用者像・基本機能・全体構成の観点から丁寧にご説明します。

---

## 🔍 論文・コードの類似性探索ツール：概要設計

---

### 🎯 **目的**

研究者や開発者が、自分の研究テーマ・技術に近い**論文・コード**を迅速かつ効率的に見つけることを目的としたツールです。

これにより：

* **関連研究**の網羅的調査が高速化される
* **既存のコード資産**を活用しやすくなる
* **自分の研究の新規性**や差別化の確認ができる
  など、研究活動の質とスピードが大幅に向上します。

---

### 🧑‍💻 **主な利用者**

* 大学院生・ポスドク
* 研究開発職（企業・アカデミア問わず）
* 技術ブログやOSSを書くエンジニア
* 研究アイデアを探している人

---

### 🛠️ **基本機能**

| 機能カテゴリ        | 詳細                                                                 |
| ------------- | ------------------------------------------------------------------ |
| 🔍 類似文献・コード検索 | 指定キーワードや文章（例：研究概要）から、類似する論文やGitHubリポジトリを検索                         |
| 📝 入力モード      | - キーワード入力<br>- テキスト（アブストラクト等）入力<br>- 自作PDFアップロード（オプション）            |
| 🧠 類似度計算      | SentenceTransformerやOpenAI Embedding APIによる意味ベクトル比較                |
| 📚 検索対象       | - 論文：arXiv API, Semantic Scholar API<br>- コード：GitHub API（+ stars順） |
| 📊 結果の可視化     | - 類似スコア付きリスト表示<br>- タグ別クラスタリング<br>- PCA/TSNEによるマップ（オプション）          |
| 📁 保存・エクスポート  | - 検索履歴の保存<br>- BibTeXやMarkdown形式で出力可能                              |

---

### 🏗️ **全体構成（システム構成概要）**

```plaintext
[入力] → [前処理] → [埋め込み生成] → [類似性評価] → [結果表示]
```

#### 1. **入力UI**

* キーワード入力欄
* 概要テキスト入力欄
* PDFアップロード（任意）

#### 2. **前処理**

* テキストクリーニング（句読点除去、英語化など）
* 段落・文単位の分割（必要に応じて）

#### 3. **埋め込み生成**

* `sentence-transformers`（例: `all-MiniLM-L6-v2`）
* または `OpenAI Embedding API`

#### 4. **検索対象データベース**

* 論文：arXiv / Semantic Scholar（API取得 or 事前DLキャッシュ）
* コード：GitHub（APIでクエリ）

#### 5. **類似性計算**

* Cosine Similarity
* 類似度ランキング化

#### 6. **結果表示UI**

* タイトル・著者・リンク・スコア表示
* ソート・フィルター機能（分野、年、スコア）
* 類似マップやキーワードクラウド（オプション）

---

### ✅ 利用例

#### 🔹ケース1：

> 「自分が考えたアルゴリズムの概要」を入れて、arXiv上で似た論文があるかを調べる。

#### 🔹ケース2：

> 「Zero-shot learning」と入力して、類似する論文と公開実装を同時に見つけたい。

#### 🔹ケース3：

> 「昨年読んだ論文に似ている最新の研究」を自動で定期チェック。

---

### 🔄 今後の拡張性（例）

* Mendeley/Zotero連携で論文管理と統合
* 論文の差分比較ツール（Diff）
* 学術トレンド分析（例：時系列クラスタリング）

---

ご希望であれば、この後に「**システム構成図（モジュール構成）**」「**開発スケジュール（WBS）**」「**プロトタイピング用サンプルコード**」などに進むこともできます。
次に知りたい情報を教えてください。
</details>

---
何を言っているのかわからない人もいるのではないでしょうか。  
でも、実装すべきものが何かを明確にしてくれているので、これを進めていってもいいアプリケーションができそうです...!!  
ただ、アイデアとしては申し分ないのですが、いかんせん工数が多すぎますね。  

## 3.出力情報の手直し
今回は簡単なやつでいいので、pythonコードだけで実装可能なものを出力してもらいましょう。  
プロンプト：  
`今回は簡単なものでいいです。
入力された指定キーワードや文章（例：研究概要）から、類似する論文をarXiv API, Semantic Scholar APIより検索し、類似スコアの高い論文10報を出力させるコードをpythonで作製しましょう。`  
ちょっと言葉は難しいかもしれませんが、提案されたものを実装するのが面倒なので、入力された情報を元にウェブスクレイピング（ネットの情報を色々持ってくること）の技術を活用して実装しようという事です。
<details><summary>折りたたみ３</summary>
了解しました。「簡単なもの」という条件に合わせて、以下のような構成で、**入力キーワードや文章からarXivとSemantic Scholarから論文を検索し、意味的に類似した上位10件を出力するPythonコード**をご提供します。

---

## ✅ 機能概要

* 入力：キーワードまたは文章（研究概要など）
* 出力：類似スコア上位10件の論文（タイトル・著者・リンク付き）
* 技術：

  * 検索：arXiv API、Semantic Scholar API
  * 類似度計算：`sentence-transformers` による埋め込み＋Cosine Similarity

---

## 🛠️ 必要ライブラリのインストール

```bash
pip install sentence-transformers requests numpy
```

---

## 🧠 Pythonコード全文

```python
import requests
import xml.etree.ElementTree as ET
import numpy as np
from sentence_transformers import SentenceTransformer, util

# --- 設定 ---
NUM_RESULTS = 20  # APIから取得する候補数
TOP_K = 10        # 出力する上位件数
MODEL_NAME = 'all-MiniLM-L6-v2'

# --- 入力 ---
query_text = input("検索したいキーワードや研究概要を入力してください：\n")

# --- sentence-transformers モデルロード ---
model = SentenceTransformer(MODEL_NAME)
query_embedding = model.encode(query_text, convert_to_tensor=True)

# --- arXiv API 検索 ---
def search_arxiv(query, max_results=NUM_RESULTS):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    response = requests.get(base_url, params=params)
    root = ET.fromstring(response.content)

    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace("\n", " ")
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        link = entry.find('{http://www.w3.org/2005/Atom}id').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
        papers.append({
            'source': 'arXiv',
            'title': title,
            'authors': authors,
            'summary': summary,
            'link': link
        })
    return papers

# --- Semantic Scholar API 検索 ---
def search_semantic_scholar(query, max_results=NUM_RESULTS):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,authors,url"
    }
    headers = {
        "User-Agent": "ResearchQOL/1.0"
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    papers = []
    for item in data.get("data", []):
        papers.append({
            'source': 'SemanticScholar',
            'title': item.get('title', ''),
            'authors': [a.get('name') for a in item.get('authors', [])],
            'summary': item.get('abstract', ''),
            'link': item.get('url', '')
        })
    return papers

# --- 類似度計算と出力 ---
def find_similar_papers(papers, query_embedding):
    summaries = [p['summary'] for p in papers]
    embeddings = model.encode(summaries, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = np.argsort(-cosine_scores.cpu().numpy())[:TOP_K]

    print(f"\n🔍 類似スコア上位 {TOP_K} 件の論文:\n")
    for i in top_indices:
        paper = papers[i]
        score = cosine_scores[i].item()
        print(f"[{paper['source']}] {paper['title']}")
        print(f"    Authors: {', '.join(paper['authors'])}")
        print(f"    Link: {paper['link']}")
        print(f"    Score: {score:.4f}\n")

# --- 実行フロー ---
arxiv_papers = search_arxiv(query_text)
sem_scholar_papers = search_semantic_scholar(query_text)
all_papers = arxiv_papers + sem_scholar_papers
find_similar_papers(all_papers, query_embedding)
```

---

## ✅ 実行結果例（例: “transformer-based summarization”）

```
検索したいキーワードや研究概要を入力してください：
transformer-based summarization for biomedical documents

🔍 類似スコア上位 10 件の論文:

[arXiv] BioBART: Pretraining and Evaluation of a Biomedical Generative Language Model
    Authors: John Smith, Jane Doe
    Link: http://arxiv.org/abs/xxxx.xxxxx
    Score: 0.8231

[SemanticScholar] ClinicalSumm: Transformer-Based Clinical Summarization
    Authors: A. Brown, T. Green
    Link: https://www.semanticscholar.org/paper/xxxx
    Score: 0.8125
...
```

---

## 📌 注意点

* Semantic Scholar API は無料・無認証での利用が可能ですが、**頻繁に叩きすぎるとレート制限されます**
* 類似度は**あくまで意味的類似性**なので、細かいアルゴリズムの違いには非対応です
* 論文数を増やす場合は、APIの仕様に注意して分割取得してください

---

## 📦 拡張候補

* PDFアップロード → テキスト抽出 → 類似検索
* 類似スコア付きのCSV出力
* GUI（Streamlitなど）

---

次に「PDFから直接検索したい」「コードの類似性も加えたい」「GUIで操作したい」など、拡張したい方向性があれば教えてください！

</details>

---
ここでコードまで出力してくれました。  
コーディングの知識がないとデバッグや意味理解は難しくなってしまいますが、なにやらよさげなコードです！
早速pythonで実装してみましょう。 

## 4.実装と出力情報
皆様にも共有しやすいようにJupyterで作製してみました。  
今回出力したコードは特にデバッグが必要なさそうですね。  
実行してみたところ、うまく作動しました!!!(出力内容の精度はアレかもですが、そういう方法なので仕方ない)  
[こちら](https://colab.research.google.com/drive/18_m1OIQqYREFbAGPhrg9ccXFGfLfbeuQ?usp=sharing) に添付しました。

入力情報は"Senescent T cell"としてみました。  
<img src="https://github.com/user-attachments/assets/bb299db6-f1e4-47fc-b3b6-be9181d15721">

すると出力は  
<img src="https://github.com/user-attachments/assets/ab702168-8587-44dc-aee6-51de1b29b4a4" width = 50%>  

こんな感じになりました！！！（出力された情報すべては載せられませんでした...）

とりあえず、一番上のやつ読んでみましたが、思ったやつとはちょっと違いましたね。  
計算生物学というか、テロメアの長さと分裂の確率モデルを構築し、細胞が不死化するのか死ぬのかの境界を明らかにした論文です。  
まあ老化に関連はしますが...  
T cellどこ行った...  

まあ今回は簡単なやつなのでこんなもんでしょう。  

＜追記＞  
類似度計算に意味的類似度を計算しているだけであり、入力情報が漠然とし過ぎているのであまり精度は高くなりません。  
今回は入力された情報や検索する論文をすべて384次元のベクトルにして、それらの距離が近いものを出力しているだけなのでどうしても思う様な結果は出ません。  
もっといいモデルがあればGoogle検索の様にいいものになるかもしれませんね。  
他にはこれを直すにはそもそものアルゴリズムを変更したり、APIから取得する候補数を増やす必要がありますね。
