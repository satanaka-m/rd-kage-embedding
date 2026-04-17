from typing import List, Dict
import networkx as nx
import numpy as np
import pickle
from kage_util import parse_glyphwiki_tsv, parse_single_line, SubGlyphComponent, KageStroke

class KageGraph(object):
    # KAGEデータをグラフ表現で管理するクラス
    # KAGEパーツの引用関係の解決に使う
    # NOTE 既存のKAGEデータと座標が一致しないもの（数ピクセル程度のズレ？）がある。strechの扱いが原因かもしれない。ただし引用関係の整理に使うだけであれば回避可能なので詳細は未検証。
    def __init__(self, kage_newest_path, kage_all_path, cids, reference_cids):
        self.data = parse_glyphwiki_tsv(kage_newest_path, kage_all_path)

        self.cid_kage_keys = self._to_kage_keys(cids)
        self.reference_kage_keys = self._to_kage_keys(reference_cids)

        self._prepare_graph()

    def _to_kage_keys(self, keys):
        if type(keys[0]) == int:
            new_keys = [self.cid2key(key) for key in keys]
        elif type(keys[0]) == str:
            new_keys = [x.replace("cid", "aj1-") for x in keys]
        else:
            raise ValueError("keys should be list of int or list of str")
        return new_keys
    
    def cid2key(self, cid: int)->str:
        return f"aj1-{cid:05d}"

    def _prepare_graph(self):
        self.graph = nx.MultiDiGraph()  # 同一KAGEパーツが複数ある場合に対応するため、MultiDiGraphを使う
        next_keys = set(self.cid_kage_keys)
        scanned_keys = set()

        while True:
            # 引用した部品を追跡するループ
            target_keys, next_keys = next_keys, set()
            for key in target_keys:
                self.graph.add_node(key, strokes=[])
                for component_description in self.data[key]:
                    component = parse_single_line(component_description)
                    if type(component) == SubGlyphComponent:
                        # 部品引用
                        if component.subglyph_id not in self.graph.nodes:
                            self.graph.add_node(component.subglyph_id, strokes=[])
                        self.graph.add_edge(key, component.subglyph_id, subglyph=component)  # エッジに部品の配置情報をもたせる
                        next_keys.add(component.subglyph_id)
                    elif type(component) == KageStroke:
                        # ストローク
                        self.graph.nodes[key]["strokes"].append(component)
            scanned_keys.update(target_keys)
            next_keys = next_keys - scanned_keys  # 既に取得済みの部品は除外
            if len(next_keys) == 0:
                break
        
        # 見本文字にエッジをつないでおく
        for ref_key in self.reference_kage_keys:
            self.graph.add_edge("_reference", ref_key)

    def get_num_parts(self, kage_key: str, part_key: str)->int:
        # 与えられたkage_keyの部品に含まれるpart_keyの数を返す
        return len(list(nx.all_simple_edge_paths(self.graph, source=kage_key, target=part_key)))

    def get_all_part_strokes(self, kage_key: str, part_key: str)->List[List[KageStroke]]:
        # 与えられたkage_keyの部品を展開してストロークのリストのリスト（同名部品のリスト）を返す
        output = []
        for i in range(self.get_num_parts(kage_key, part_key)):
            output.append(self.get_strokes(kage_key, part_key, i))
        return output
           
    def get_strokes(self, cid_key: str, part_key: str="", part_idx: int=0)->List[KageStroke]:
        # 与えられたcid_keyの部品を展開してストロークのリストを返す
        # part_keyが指定されている場合は、その部品より下のみを返す（引用変換はすべて適用する）
        # part_keyに該当する部品が複数ある場合は、part_idxで指定する
        pivot_key = part_key if len(part_key)>0 else cid_key
        above_paths = list(nx.all_simple_edge_paths(self.graph, source=cid_key, target=pivot_key))  # pivot_keyより上のパスが複数ある場合は、pivot_keyのパーツが複数あることを意味する
        above_path = above_paths[part_idx]

        output_strokes = []
        for key in nx.descendants(self.graph, pivot_key) | {pivot_key}:
            # 親ノードから子ノードの順で、パーツ以外の独立したストロークがあれば追加していく
            part = self.graph.nodes[key]
            if len(part["strokes"])>0:
                # 変換を適用する
                below_paths = list(nx.all_simple_edge_paths(self.graph, source=pivot_key, target=key))
                for below_path in below_paths:
                    path = above_path + below_path
                    strokes = part["strokes"]
                    for edge_idx in path[::-1]:  # 子要素から親要素の順で変換
                        strokes = self.graph.edges[edge_idx]["subglyph"].transform(strokes)
                    output_strokes.extend(strokes) 
        return output_strokes   

    def get_reference_parts(self, kage_key: str)->Dict[str, List[str]]:
        # 与えられたkage_keyの各部品について、部品を共有している見本文字を返す
        # 見本文字に部品が見つかった場合は、それより下の部品は無視する

        output = {}  # {(kage_part_key, part_idx): [reference_key1, reference_key2, ...]}
        skip_nodes = set()
        for target_part in nx.bfs_tree(self.graph, source=kage_key):
            if len(skip_nodes & set(self.graph.predecessors(target_part))) > 0:
                # 注目ノードの親ノードがスキップ対象の場合
                skip_nodes.add(target_part)
                continue
            paths = [tuple(p) for p in nx.all_simple_paths(self.graph, source="_reference", target=target_part)]
            paths = list(set(paths))  # リファレンス側に同一部品がある場合に重複するため重複除去
            for path in paths:
                for node in path:
                    if node in self.reference_kage_keys:  # 見本文字のとき
                        skip_nodes.add(target_part)
                        if target_part not in output:
                            output[target_part] = [node]
                        else:
                            output[target_part].append(node)
        return output

    # 見本文字にパーツが見つからなかったストロークを返す
    def get_unresolved_strokes(self, kage_key: str)->List[KageStroke]:
        resolved_part_keys = self.get_reference_parts(kage_key).keys()
        unresolved_strokes = self.get_strokes(kage_key)  # 初期化
        for resolved_part_key in resolved_part_keys:
            for strokes in self.get_all_part_strokes(kage_key, resolved_part_key):
                for stroke in strokes:
                    if stroke in unresolved_strokes:
                        unresolved_strokes.remove(stroke)
        return unresolved_strokes
        
    def get_subnodes(self, kage_key: str)->List[str]:
        # 特定CID以下のサブノードを取得
        nodes = nx.descendants(self.graph, kage_key) | {kage_key}
        return list(nodes)

    def get_subgraph(self, kage_key: str)->nx.MultiDiGraph:
        # 特定CID以下のサブグラフを取得
        nodes = self.get_subnodes(kage_key)
        return self.graph.subgraph(nodes).copy()
   
    def get_component_series(self, kage_key: str) -> List[List[str]]:
        # kage_keyの部品の系列を取得
        output = []
        subgraph = self.get_subgraph(kage_key)
        for node in subgraph.nodes:
            ancestors_path = nx.all_simple_paths(subgraph, source=kage_key, target=node)
            ancestors = list(ancestors_path)
            if len(ancestors) > 1:  # 複数の同一部品がある場合など
                assert(all([len(ancestors[0])==len(ancestor) for ancestor in ancestors]))  # すべての要素が同一であることを確認
            ancestors = list(ancestors[0])
            output.append(ancestors)
        return output
 

# main
if __name__ == "__main__":
    from kage_util import strokes_to_img
    from PIL import Image
    from cid_table import CID_TABLES, GLYPHSET
    cids = CID_TABLES[GLYPHSET.MIN2_KANJI]
    reference_cids = CID_TABLES[GLYPHSET.EXP_SELECT]
    kage_graph = KageGraph("data/kage_dump/dump_newest_only.txt", "data/kage_dump/dump_all_versions.txt", cids, reference_cids)

    # 一部パーツのみを表示してみる
    # kage_key = "aj1-03516"  # 品（複数の同一部品を含む例）
    # kage_key = "aj1-06925"  # 邂（ストレッチを含む例）
    # kage_key = "aj1-01601"  # 祈
    for kage_key in ["aj1-03516", "aj1-06925", "aj1-01601"]:

        # 特定CID以下のサブグラフを取得
        ancestor_series = kage_graph.get_component_series(kage_key)
        for series in ancestor_series:
            parts_string = "_".join(series)
            for i, strokes in enumerate(kage_graph.get_all_part_strokes(kage_key, series[-1])):
                img = strokes_to_img(strokes)
                Image.fromarray(img).save(f"output/series_{parts_string}_{i}.png")


        # グラフの可視化
        # subgraph = kage_graph.get_subgraph(kage_key)
        # import matplotlib.pyplot as plt
        # pos = nx.spring_layout(subgraph)
        # nx.draw(subgraph, pos)
        # nx.draw_networkx_labels(subgraph, pos)
        # plt.show()

        # 対応する見本文字を表示
        references = kage_graph.get_reference_parts(kage_key)
        img = strokes_to_img(kage_graph.get_strokes(kage_key))
        for part_key in references:
            components = kage_graph.get_all_part_strokes(kage_key, part_key)
            imgs_part = [strokes_to_img(components[i]) for i in range(len(components))]

            # 見本文字
            imgs_ref = []
            for reference_key in references[part_key]:
                img_ref = strokes_to_img(kage_graph.get_strokes(reference_key))
                components = kage_graph.get_all_part_strokes(reference_key, part_key)
                imgs_ref_part = [strokes_to_img(components[i]) for i in range(len(components))]
                imgs_ref.append(img_ref)
                imgs_ref.extend(imgs_ref_part)
            if len(imgs_ref) > 0:
                concat_img = np.concatenate([img] + imgs_part + imgs_ref, axis=1)
                Image.fromarray(concat_img).save(f"output/matching_{kage_key}_{part_key}.png")

        # パーツと未割り当てストロークを並べて表示
        img_strokes = strokes_to_img(kage_graph.get_unresolved_strokes(kage_key))
        concat_img = np.concatenate([img, img_strokes], axis=1)
        Image.fromarray(concat_img).save(f"output/parts_strokes_{kage_key}.png")

