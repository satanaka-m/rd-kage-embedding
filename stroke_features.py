# KAGEデータのタイプや方向などを考慮したストローク比較のための関数

from typing import List, Tuple, Dict
import numpy as np
from kage_util import KageStroke, StrokeType, StartPointType, EndPointType

# 3次ベジエ曲線の曲率計算
def curvature(t, P0, P1, P2, P3):
    B_prime = 3 * (1 - t) ** 2 * (P1 - P0) + 6 * (1 - t) * t * (P2 - P1) + 3 * t ** 2 * (P3 - P2)
    B_double_prime = 6 * (1 - t) * (P2 - 2 * P1 + P0) + 6 * t * (P3 - 2 * P2 + P1)
    
    # 2D cross product (gives scalar value)
    cross_product = B_prime[0] * B_double_prime[1] - B_prime[1] * B_double_prime[0]
    
    # Norm (magnitude) of B'(t)
    norm_B_prime = np.linalg.norm(B_prime)
    
    # Signed Curvature
    return cross_product / (norm_B_prime ** 3)

def same_stroke_groups(stroke1: KageStroke, stroke2: KageStroke) -> bool:
    stroke_type_groups = [[StrokeType.LINE], [StrokeType.QCURVE], [StrokeType.CCURVE], [StrokeType.HARAI], [StrokeType.ORE], [StrokeType.OTSU]]
    group_idx1 = [stroke1.stroke_type in tg for tg in stroke_type_groups].index(True)
    group_idx2 = [stroke2.stroke_type in tg for tg in stroke_type_groups].index(True)
    return group_idx1 == group_idx2

def similar_stroke_groups(stroke1: KageStroke, stroke2: KageStroke) -> bool:
    ## QCURVE, CCURVE, HARAIは同じとみなす
    stroke_type_groups = [[StrokeType.LINE], [StrokeType.QCURVE, StrokeType.CCURVE, StrokeType.HARAI], [StrokeType.ORE], [StrokeType.OTSU]]
    group_idx1 = [stroke1.stroke_type in tg for tg in stroke_type_groups].index(True)
    group_idx2 = [stroke2.stroke_type in tg for tg in stroke_type_groups].index(True)
    return group_idx1 == group_idx2

def same_startpoint_groups(stroke1: KageStroke, stroke2: KageStroke) -> bool:
    ## HOSOIRIとYANE_HOSOIRIは同じとみなす
    ## TODO: OPENとCONNECT_Hは同一視したほうがよいか確認
    startpointtype_groups = [[StartPointType.OPEN], [StartPointType.CONNECT_H], [StartPointType.CONNECT_V], [StartPointType.KADO_UL], [StartPointType.KADO_UR], [StartPointType.HOSOIRI, StartPointType.YANE_HOSOIRI]]
    group_idx1 = [stroke1.start_type in tg for tg in startpointtype_groups].index(True)
    group_idx2 = [stroke2.start_type in tg for tg in startpointtype_groups].index(True)
    return group_idx1 == group_idx2

def same_endpoint_groups(stroke1: KageStroke, stroke2: KageStroke) -> bool:
    ## ZH_DL_OLDとZH_DL_NEWは同じとみなす
    endpointtype_groups = [[EndPointType.OPEN], [EndPointType.CONNECT_H], [EndPointType.CONNECT_V], [EndPointType.KADO_DL], [EndPointType.KADO_DR], [EndPointType.HANE], [EndPointType.ZH_DL_OLD, EndPointType.ZH_DL_NEW], [EndPointType.KADO_DR_HT], [EndPointType.HARAI], [EndPointType.TOME], [EndPointType.HANE_U]]
    group_idx1 = [stroke1.end_type in tg for tg in endpointtype_groups].index(True)
    group_idx2 = [stroke2.end_type in tg for tg in endpointtype_groups].index(True)
    return group_idx1 == group_idx2

def same_curvature_sign(stroke1: KageStroke, stroke2: KageStroke) -> bool:
    ## 中間地点（ベジェ媒介変数0.5）での曲率の符号が同じかどうかを比較
    ## 符号付き曲率の符号が途中で変わるケースはあるが、（AJ1漢字の範囲では）ごく小さなものなので無視する。
    ## HARAIの曲がり方向は一種類だけなので比較不要
    if stroke1.stroke_type in [StrokeType.QCURVE, StrokeType.CCURVE]:
        p1 = stroke1.get_overcomplete_controls()
        c1 = curvature(0.5, p1[0], p1[1], p1[2], p1[3])
        p2 = stroke2.get_overcomplete_controls()
        c2 = curvature(0.5, p2[0], p2[1], p2[2], p2[3])
        return c1 * c2 > 0
    else:
        return True

def similar_direction(stroke1: KageStroke, stroke2: KageStroke, angle_threshold) -> bool:
    vec1 = stroke1.ctrls[-1] - stroke1.ctrls[0]
    a1 = np.arctan2(vec1[1], vec1[0])
    vec2 = stroke2.ctrls[-1] - stroke2.ctrls[0]
    a2 = np.arctan2(vec2[1], vec2[0])

    angle_diff = np.rad2deg(np.abs(a1 - a2))
    return angle_diff < angle_threshold

def stroke_stroke_penalty(stroke1: KageStroke, stroke2: KageStroke, group_curves=True) -> float:
    # タイプや図形的な特徴の異なるストロークに対するペナルティ

    different_type_penalty = 100
    different_pointtype_penalty = 10
    different_curvature_sign_penalty = 100
    different_angle_penalty = 100

    different_angle_threshold = 30

    distance = 0

    # StrokeTypeによる分類比較
    ## 座標や方向の比較は別途行う
    if group_curves:
        if not similar_stroke_groups(stroke1, stroke2):
            distance += different_type_penalty
    else:
        if not same_stroke_groups(stroke1, stroke2):
            distance += different_type_penalty

    # StartPointTypeによる分類比較
    if not same_startpoint_groups(stroke1, stroke2):
        distance += different_pointtype_penalty
        
    # EndPointTypeによる分類比較
    if not same_endpoint_groups(stroke1, stroke2):
        distance += different_pointtype_penalty
    
    # 曲がり方向の比較
    if not same_curvature_sign(stroke1, stroke2):
        distance += different_curvature_sign_penalty

    # 方向の比較
    if not similar_direction(stroke1, stroke2, angle_threshold=different_angle_threshold):
        distance += different_angle_penalty

    return distance

def stroke_stroke_distance(stroke1: KageStroke, stroke2: KageStroke) -> float:
    # ストロークのサイズや位置をもとにした類似度を計算（精度はよくない）
    distance = 0

    # タイプ・方向などの比較（ペナルティ項）
    distance += stroke_stroke_penalty(stroke1, stroke2)

    # サイズの比較
    z = 1/200
    bbox1 = stroke1.bbox()
    w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    bbox2 = stroke2.bbox()
    w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    distance += z * (np.abs(w1 - w2) + np.abs(h1 - h2))

    # 縦位置または横位置が近いものを優先
    # 始筆部と終筆部の平均で比較
    z = 1/200
    p1 = stroke1.ctrls[[0, -1]].mean(axis=0)
    p2 = stroke2.ctrls[[0, -1]].mean(axis=0)
    dx = z * np.abs(p1[0] - p2[0])
    dy = z * np.abs(p1[1] - p2[1])
    # distance += min([dx, dy])
    beta = 2.0
    distance += ( dx*np.exp(-beta*dx) + dy*np.exp(-beta*dy) ) / (np.exp(-beta*dx) + np.exp(-beta*dy))  # 近い方の距離を重視して重み付け

    return distance


def combination_distances(target_glyph: List[KageStroke], reference_glyphs: List[List[KageStroke]]) -> np.ndarray:
    # 各組み合わせの距離を計算
    max_reference_strokes = max([len(rg) for rg in reference_glyphs])  # 見本文字の最大ストローク数
    distances = np.zeros((len(target_glyph), len(reference_glyphs), max_reference_strokes)) + 1e6  # ストローク数が文字ごとに違うため未使用領域ができる。未使用領域には大きな値を入れておく。
    for i, target_stroke in enumerate(target_glyph):
        for j, reference_glyph in enumerate(reference_glyphs):
            for k, reference_stroke in enumerate(reference_glyph):
                distances[i, j, k] = stroke_stroke_distance(target_stroke, reference_stroke)
    return distances


def simple_assign(target_glyph: List[KageStroke], reference_glyphs: List[List[KageStroke]]) -> List[Tuple[int, int, int, float]]:
    """
    各ターゲットストロークごとに、最も類似するリファレンス文字とストロークを探して割り当て（重複あり）

    Args:
        target_glyph(List[KageComponent]): ターゲット文字のストロークリスト
        reference_glyphs(List[List[KageComponent]]): 見本文字のストロークリストのリスト

    Returns:
        List[Tuple[int, int, int, float]]: 割り当てのリスト。各割り当ては[ターゲットストロークインデックス, リファレンス文字インデックス, リファレンスストロークインデックス, 距離]で表現される。
    """
    distances = combination_distances(target_glyph, reference_glyphs)

    # 最も近いストロークを探す
    result = []
    for i in range(len(target_glyph)):
        j, k = np.unravel_index(np.argmin(distances[i]), distances[i].shape)
        distance = distances[i, j, k]
        result.append([i, j, k, distance])
    return result
