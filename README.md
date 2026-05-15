***WIP：要パラメータ調整***

KAGEデータと黎ミンの画像でSigRegをやってみる実験


# 背景

KAGEデータは表現の一貫性が乏しい（同じ部首でも別の表現になるなど）ため、画像と関連付けることで表現の一貫性を高めることや、エラー修正（自動・手動）ができるかもしれない

# 目的

下記の実現を目指す
- 一貫性があること：ほぼ同一字形をあらわす別々の表現が同じ形式にまとまることを期待
- 同一部首の判定ができること
- パーツ包含の判定ができること（例：一寸の巾）
- 類似字形が縮退しないこと

# 手法

KAGEデータは最小のストローク単位に分解し、エンコード層＋Transformer、画像側はDINOv2で特徴量抽出し、対応ペアのMSEとSigRegで両者の特徴量を関連付けるよう学習させる。

SigReg は `conf/config.yaml` の `loss.variant` で切り替える。

- `weak`: 共分散を単位行列へ寄せる Weak-SIGReg
- `strong`: LeJEPA 公式実装に近い sliced Epps-Pulley の Strong-SIGReg


# 発展

- ゴシック体もmulti-view設定で扱えるようにする
- IDSとの関連付けをやってみる
- パーツ表現のKGAEも埋め込めるようにする

# データ

KAGEデータのdumpは以下からダウンロードできる
https://glyphwiki.org/dump.tar.gz

ただし 2026-04 時点では、デフォルトの `wget` / `curl` は Cloudflare の bot 判定により `403 Forbidden` になることがある。
リモート CLI から取得する場合は、ブラウザ相当の User-Agent を付けてダウンロードする。

```bash
mkdir -p data/kage_dump

wget \
  --user-agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36' \
  -O /tmp/dump.tar.gz \
  https://glyphwiki.org/dump.tar.gz

tar -xzf /tmp/dump.tar.gz -C data/kage_dump
```

`curl` を使う場合:

```bash
mkdir -p data/kage_dump

curl -L \
  -A 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36' \
  https://glyphwiki.org/dump.tar.gz \
  -o /tmp/dump.tar.gz

tar -xzf /tmp/dump.tar.gz -C data/kage_dump
```

それでも取得できない場合は、ブラウザで上記 URL を開いてダウンロードし、展開後のファイルを `data/kage_dump` に配置する。

必要なファイル:
- `data/kage_dump/dump_newest_only.txt`
- `data/kage_dump/dump_all_versions.txt`

例:

```bash
mkdir -p data/kage_dump
tar -xzf ~/Downloads/dump.tar.gz -C data/kage_dump
```

# 使用方法

## MLflow運用ルール（推奨）

デバッグ用途を除き、`mlflow run` では `--experiment-name` と `--run-name` を必ず指定する。  
未指定だと `Default` experiment に記録され、run名も自動生成されるため、後で追跡しにくくなる。

## 1. KAGEデータのパース

KAGEのダンプファイルをパースしてpickleファイルに変換：

```bash
# 推奨: experiment / run 名を明示
mlflow run . -e parse_kage \
  --experiment-name "sigreg-data-prep" \
  --run-name "parse-kage-20260403"
```

KAGE ダンプの場所を明示する場合:

```bash
mlflow run . -e parse_kage \
  --experiment-name "sigreg-data-prep" \
  --run-name "parse-kage-custom-20260403" \
  -P kage_dump_dir=./data/kage_dump
```

## 2. データの確認

パースされたデータを確認・可視化：

```bash
# 推奨: experiment / run 名を明示
mlflow run . -e check_data \
  --experiment-name "sigreg-data-check" \
  --run-name "check-data-20260403"
```

ストローク SVG も出す場合:

```bash
mlflow run . -e check_data \
  --experiment-name "sigreg-data-check" \
  --run-name "check-data-with-strokes-20260403" \
  -P output_strokes=true
```

## 3. モデルの訓練

SigRegモデルの訓練：

現状のデフォルト設定は `Strong-SIGReg`。`Weak-SIGReg` を使う場合は、事前に [conf/config.yaml](conf/config.yaml) の `loss.variant` を `weak` に変える。

主な損失パラメータ:

- `loss.variant`: `weak` または `strong`
- `loss.sim_weight`: 対応ペア MSE の重み
- `loss.reg_weight`: SIGReg 正則化の重み
- `loss.sketch_dim`: Weak-SIGReg の sketch 後の次元
- `loss.num_slices`: Strong-SIGReg のランダム 1D 射影数
- `loss.epps_pulley_t_max`: Strong-SIGReg の Epps-Pulley 積分上限
- `loss.epps_pulley_num_points`: Strong-SIGReg の Epps-Pulley 積分点数

`mlflow run` はこのリポジトリでは [python_env.yaml](python_env.yaml) を使って専用の virtualenv を作る。
そのため、CUDA 関連のエラーはホスト OS ではなく、その MLflow 仮想環境に入った PyTorch wheel の組み合わせが原因になりやすい。

このリポジトリでは GPU 実行用に [requirements.txt](requirements.txt) を CUDA 12.8 向けへ固定している。

- `torch==2.8.0`
- `torchvision==0.23.0`
- `--index-url https://download.pytorch.org/whl/cu128`

`RuntimeError: The NVIDIA driver on your system is too old (found version 12080)` は、MLflow 仮想環境側でホストの driver より新しい CUDA 向け PyTorch が入ったときに出る。
その場合は依存を固定したうえで、次のように GPU 実行する。

```bash
mlflow run . -e train \
  --experiment-name "sigreg-training" \
  --run-name "train-baseline-20260403"
```

以前に作られた不整合な MLflow 環境が再利用される場合だけ、キャッシュを消して作り直す。

```bash
rm -rf ~/.mlflow/envs
```

実装上、学習時のデバイスは自動選択される。CUDA が使える環境では GPU、使えない環境では CPU になる。

```python
accelerator='gpu' if torch.cuda.is_available() else 'cpu'
```

そのため、次のコマンドは「CPU を強制する」ものではなく、CUDA が見えない環境であれば CPU 実行になる通常の学習コマンドである。

```bash
mlflow run . -e train \
  --experiment-name "sigreg-training" \
  --run-name "train-cpu-20260403"
```

カスタム実験名とパラメータを指定:

```bash
mlflow run . -e train \
  --experiment-name "sigreg-experiment-v1" \
  --run-name "run-20260403"
```

学習設定は [conf/config.yaml](conf/config.yaml) を編集して調整する。MLproject 経由では学習パラメータを上書きしない。

```bash
mlflow run . -e train \
  --experiment-name "sigreg-training" \
  --run-name "train-from-config-20260403" \
  -P config=conf/config.yaml
```

設定ファイル変更後に MLflow から実行する例:

```bash
mlflow run . -e train \
  --experiment-name "sigreg-training" \
  --run-name "train-from-config-20260403"
```

## 4. 学習済みモデルの k近傍可視化

学習済みチェックポイントから埋め込みを計算し、cos類似度で近傍検索した以下の2方向を可視化する。

- `glyph -> top K shape`
- `shape -> top K glyph`

字形側は見た目だけだと解釈しにくいため、各候補にグリフ画像と `CID` を併記して PNG として保存する。

学習後のチェックポイントは `checkpoints/...` には自動保存されず、`models/<MLflow run_id>/model.ckpt` に保存される。

```bash
python visualize_knn.py \
  --config conf/config.yaml \
  --checkpoint models/<MLflow run_id>/model.ckpt \
  --output-dir outputs/knn \
  --top-k 20 \
  --num 20 \
  --batch-size 64 \
  --seed 42
```

`mlflow run` から実行する場合:

```bash
mlflow run . -e visualize_knn \
  -P checkpoint=models/<MLflow run_id>/model.ckpt \
  -P output_dir=outputs/knn \
  -P top_k=20 \
  -P num=20 \
  -P batch_size=64 \
  -P seed=42
```

## MLflow UI での結果確認

```bash
mlflow ui
```

ブラウザで http://localhost:5000 にアクセスして実験結果を確認できます
