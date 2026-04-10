***WIP：要パラメータ調整***

KAGEデータと黎ミンの画像でVICRegをやってみる実験


# 背景

KAGEデータは表現の一貫性が乏しい（同じ部首でも別の表現になるなど）ため、画像と関連付けることで表現の一貫性を高めることや、エラー修正（自動・手動）ができるかもしれない

# 目的

下記の実現を目指す
- 一貫性があること：ほぼ同一字形をあらわす別々の表現が同じ形式にまとまることを期待
- 同一部首の判定ができること
- パーツ包含の判定ができること（例：一寸の巾）
- 類似字形が縮退しないこと

# 手法

KAGEデータは最小のストローク単位に分解し、エンコード層＋Transformer、画像側はDINOv2で特徴量抽出し、VICRegで両者の特徴量を関連付けるよう学習させる


# 発展

- ゴシック体もmulti-view設定で扱えるようにする
- IDSとの関連付けをやってみる
- パーツ表現のKGAEも埋め込めるようにする

# データ

KAGEデータのdumpは以下からダウンロードできる
https://glyphwiki.org/dump.tar.gz

# 使用方法

## MLflow運用ルール（推奨）

デバッグ用途を除き、`mlflow run` では `--experiment-name` と `--run-name` を必ず指定する。  
未指定だと `Default` experiment に記録され、run名も自動生成されるため、後で追跡しにくくなる。

## 1. KAGEデータのパース

KAGEのダンプファイルをパースしてpickleファイルに変換：

```bash
# 推奨: experiment / run 名を明示
mlflow run . -e parse_kage \
  --experiment-name "vicreg-data-prep" \
  --run-name "parse-kage-20260403"

# カスタムパスを指定
mlflow run . -e parse_kage \
  --experiment-name "vicreg-data-prep" \
  --run-name "parse-kage-custom-20260403" \
  -P kage_dump_dir=./data/kage_dump \
  -P kage_pkl=./data/kage.pkl
```

## 2. データの確認

パースされたデータを確認・可視化：

```bash
# 推奨: experiment / run 名を明示
mlflow run . -e check_data \
  --experiment-name "vicreg-data-check" \
  --run-name "check-data-20260403"

# カスタム設定ファイルを使用
mlflow run . -e check_data \
  --experiment-name "vicreg-data-check" \
  --run-name "check-data-custom-config-20260403" \
  -P config=conf/config.yaml
```

## 3. モデルの訓練

VICRegモデルの訓練：

```bash
# 推奨: train entry point内の引数と、mlflow run側の指定をそろえる
mlflow run . -e train \
  --experiment-name "vicreg-training" \
  --run-name "train-baseline-20260403" \

# カスタム実験名とパラメータを指定
mlflow run . -e train \
  --experiment-name "vicreg-experiment-v1" \
  --run-name "run-20260403" \
  -P experiment_name="vicreg-experiment-v1" \
  -P run_name="run-20260403"

# 訓練パラメータをカスタマイズ
mlflow run . -e train \
  -P experiment_name="vicreg-training" \
  -P run_name="train-custom-20260403" \
  -P max_epochs=200 \
  -P batch_size=64 \
  -P learning_rate=5.0e-5 \
  -P dim_emb=512 \
  -P tf_dim_model=256 \
  -P tf_dim_ff=512

# 複数のパラメータを組み合わせた例
mlflow run . -e train \
  -P experiment_name="vicreg-tuning" \
  -P run_name="tuning-lr-search" \
  -P max_epochs=100 \
  -P batch_size=32 \
  -P learning_rate=1.0e-4 \
  -P dim_emb=256 \
  -P tf_dim_model=256 \
  -P tf_dim_ff=256
```

## 4. 学習済みモデルの k近傍可視化

学習済みチェックポイントから埋め込みを計算し、cos類似度で近傍検索した以下の2方向を可視化する。

- `glyph -> top K shape`
- `shape -> top K glyph`

字形側は見た目だけだと解釈しにくいため、各候補にグリフ画像と `CID` を併記して PNG として保存する。

```bash
python visualize_knn.py \
  --config conf/config.yaml \
  --checkpoint checkpoints/epoch=2-step=1239.ckpt \
  --output-dir outputs/knn \
  --top-k 20 \
  --num 20 \
  --batch-size 64 \
  --seed 42
```

`mlflow run` から実行する場合:

```bash
mlflow run . -e visualize_knn \
  -P checkpoint=checkpoints/epoch=2-step=1239.ckpt \
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
