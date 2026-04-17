import argparse
import pickle
import yaml
from pathlib import Path
from cid_table import CID_TABLES, GLYPHSET
from kage_util import parse_glyphwiki_tsv, extract_aj1_related, parse_components, expand_glyph
import mlflow

def load_config(config_file):
    """YAML設定ファイルをロードする"""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # 設定をロード
    config = load_config(args.config)
    
    # コマンドライン引数で上書き
    kage_dir = args.kage_dump_dir or config['dataset']['kage_dump_dir']
    pkl_path = args.kage_pkl or config['dataset']['kage_pkl']
    
    # mlflowで記録開始（既存ランがない場合のみ）
    is_run_started = mlflow.active_run() is not None
    if not is_run_started:
        mlflow.start_run()
    
    try:
        # パラメータを記録
        mlflow.log_param("kage_dump_dir", kage_dir)
        mlflow.log_param("kage_pkl", pkl_path)
        
        if Path(pkl_path).exists():
            print("file exists:", pkl_path)
            print("skip prepare process.")
            mlflow.log_param("status", "skipped")
        else:
            cids = CID_TABLES[GLYPHSET.PR6N_KANJI]
            # prepare pkl
            newest_path = Path(kage_dir) / "dump_newest_only.txt"
            all_versions_path = Path(kage_dir) / "dump_all_versions.txt"
            
            print(f"Processing KAGE data from {kage_dir}")
            data = parse_glyphwiki_tsv(newest_path.resolve(), all_versions_path.resolve())
            print(f"Total glyphs before filtering: {len(data)}")
            
            data = extract_aj1_related(data, cids)
            print(f"Total glyphs after AJ1 filtering: {len(data)}")
            
            # 指定したCIDが全部含まれているか確認
            missing = set(map(lambda x: f"aj1-{x:05d}", cids)).difference(set(data.keys()))
            assert len(missing) == 0, f"Missing CIDs: {missing}"
            glyphs = parse_components(data)
            
            expanded_glpyhs = {}
            for x in cids:
                key = f"aj1-{x:05d}"
                expanded_glpyhs[key] = expand_glyph(glyphs[key], glyphs)
            
            # 出力ディレクトリを作成
            Path(pkl_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(pkl_path, "wb") as f:
                pickle.dump(expanded_glpyhs, f)
            
            print(f"Saved expanded glyphs to {pkl_path}")
            mlflow.log_artifact(pkl_path, "data")
            mlflow.log_param("status", "completed")
            mlflow.log_metric("num_glyphs", len(expanded_glpyhs))
    finally:
        if not is_run_started:
            mlflow.end_run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse KAGE TSV data")
    parser.add_argument("--config", type=str, default="conf/config.yaml",
                        help="Path to config file")
    parser.add_argument("--kage_dump_dir", type=str, default=None,
                        help="Path to KAGE dump directory")
    parser.add_argument("--kage_pkl", type=str, default=None,
                        help="Path to output pickle file")
    
    args = parser.parse_args()
    main(args)