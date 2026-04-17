import copy
import numpy as np
from enum import Enum
from typing import List

class StrokeType(Enum):
    LINE = 1
    QCURVE = 2
    ORE = 3
    OTSU = 4
    CCURVE = 6
    HARAI = 7
    SUBGLYPH = 99
    SPECIAL = 0

    @classmethod
    def _missing_(cls, value):
        return cls(value % 100)

class StartPointType(Enum):
    OPEN = 0
    CONNECT_H = 2
    CONNECT_V = 32
    KADO_UL = 12
    KADO_UR = 22
    HOSOIRI = 7
    YANE_HOSOIRI = 27

    @classmethod
    def _missing_(cls, value):
        return cls(value % 100)

class EndPointType(Enum):
    OPEN = 0
    CONNECT_H = 2
    CONNECT_V = 32
    KADO_DL = 13
    KADO_DR = 23
    HANE = 4
    ZH_DL_OLD = 313
    ZH_DL_NEW = 413
    KADO_DR_HT = 24
    HARAI = 7
    TOME = 8
    HANE_U = 5  # R/U

    @classmethod
    def _missing_(cls, value):
        v = value if value in [313, 413] else value % 100
        return cls(v)

def normalize_delta(xd, yd, kmage):
    l = (xd**2 + yd**2)**0.5
    return xd*kmage/l, yd*kmage/l

class KageStroke(object):
    # KAGEのストロークを表現するクラス
    # TODO 実態としてはKageストロークを表現するだけなので、名前を変更する

    def __init__(self, stroke_type: StrokeType, start_type: StartPointType, end_type: EndPointType, ctrls):
        self.stroke_type = stroke_type
        self.start_type = start_type
        self.end_type = end_type
        self.ctrls = np.array(ctrls)
        
    def bbox(self):
        xs, ys = self.ctrls[:, 0], self.ctrls[:, 1]
        return [min(xs), min(ys), max(xs), max(ys)]

    def get_stroketype_idx(self):
        return list(StrokeType).index(self.stroke_type)
    
    def get_startpointtype_idx(self):
        return list(StartPointType).index(self.start_type)

    def get_endpointtype_idx(self):
        return list(EndPointType).index(self.end_type)

    def get_overcomplete_controls(self):
        if self.stroke_type == StrokeType.LINE:
            p0, p1 = self.ctrls
            return np.array([p0, p0*2/3+p1*1/3, p0*1/3+p1*2/3, p1])/200
        elif self.stroke_type == StrokeType.QCURVE:
            p0, p1, p2 = self.ctrls
            return np.array([p0, p0*1/3+p1*2/3, p1*2/3+p2*1/3, p2])/200
        elif self.stroke_type == StrokeType.ORE:
            p0, p1, p2 = self.ctrls
            return np.array([p0, p1, p1, p2])/200
        elif self.stroke_type == StrokeType.OTSU:
            p0, p1, p2 = self.ctrls
            return np.array([p0, p1, p1, p2])/200
        elif self.stroke_type == StrokeType.CCURVE:
            p0, p1, p2, p3 = self.ctrls
            return np.array([p0, p1, p2, p3])/200
        elif self.stroke_type == StrokeType.HARAI:
            p0, p1, p2, p3 = self.ctrls
            return np.array([p0, p1, p2, p3])/200
        else:
            raise ValueError("invalid stroke type")

    def __eq__(self, other):
        return self.stroke_type == other.stroke_type and self.start_type == other.start_type and self.end_type == other.end_type and np.all(self.ctrls == other.ctrls)

    def __str__(self):
        return ",".join(map(str, [self.stroke_type, self.start_type, self.end_type]+self.ctrls.flatten().tolist()))

    def svg_path(self):
        if self.stroke_type == StrokeType.LINE:
            # 直線
            x0, y0, x1, y1 = self.ctrls.flatten().tolist()
            return f"M {x0} {y0} L {x1} {y1}"
        elif self.stroke_type == StrokeType.QCURVE:
            # 曲線
            x0, y0, x1, y1, x2, y2 = self.ctrls.flatten().tolist()
            return f"M {x0} {y0} Q {x1} {y1}, {x2} {y2}"
        elif self.stroke_type == StrokeType.ORE:
            # 折れ
            x0, y0, x1, y1, x2, y2 = self.ctrls.flatten().tolist()
            return f"M {x0} {y0} L {x1} {y1} L {x2} {y2}"
        elif self.stroke_type == StrokeType.OTSU:
            # 乙線
            x0, y0, x1, y1, x2, y2 = self.ctrls.flatten().tolist()
            return f"M {x0} {y0} L {x1} {y1} L {x2} {y2}"
        elif self.stroke_type == StrokeType.CCURVE:
            # 複曲線
            x0, y0, x1, y1, x2, y2, x3, y3 = self.ctrls.flatten().tolist()
            return f"M {x0} {y0} C {x1} {y1}, {x2} {y2}, {x3} {y3}"
        elif self.stroke_type == StrokeType.HARAI:
            # 縦払い
            x0, y0, x1, y1, x2, y2, x3, y3 = self.ctrls.flatten().tolist()
            return f"M {x0} {y0} L {x1} {y1} Q {x2} {y2}, {x3} {y3}"
        else:
            raise ValueError

KageComponent = KageStroke  # 過去保存したpklを読み出すためのエイリアス

class SubGlyphComponent(object):
    def __init__(self, subglyph_id, xd, yd, xs, ys, x0, y0, x1, y1):
        self.subglyph_id = subglyph_id
        self.xd, self.yd, self.xs, self.ys = xd, yd, xs, ys
        self.bbox = [x0, y0, x1, y1]

    def __str__(self):
        return ",".join(map(str, [self.subglyph_id, self.xd, self.yd, self.xs, self.ys, self.bbox]))
    
    def transform(self, strokes: List[KageStroke]):
        strokes = copy.deepcopy(strokes)
        # bbox計算
        bbox = [200, 200, 0, 0] # 必ずアップデートされるような値で初期化
        for stroke in strokes:
            b = stroke.bbox()
            bbox[0] = min(bbox[0], b[0])
            bbox[1] = min(bbox[1], b[1])
            bbox[2] = max(bbox[2], b[2])
            bbox[3] = max(bbox[3], b[3])
        assert(bbox[0]<200 and bbox[1]<200 and 0<bbox[2] and 0<bbox[3])
    
        # 座標変換
        xd, yd, xs, ys = self.xd, self.yd, self.xs, self.ys
        x0, y0, x1, y1 = self.bbox
        if not (xd == yd == xs == ys == 0):
            # stretchありの場合
            for stroke in strokes:
                for i in range(stroke.ctrls.shape[0]):
                    x, y = stroke.ctrls[i]
                    xt = stretch(xd, xs, x, bbox[0], bbox[2])
                    yt = stretch(yd, ys, y, bbox[1], bbox[3])
                    stroke.ctrls[i, 0] = xt*(x1-x0)/200 + x0
                    stroke.ctrls[i, 1] = yt*(y1-y0)/200 + y0
        else:
            for stroke in strokes:
                stroke.ctrls[:, 0] = stroke.ctrls[:, 0]*(x1-x0)/200 + x0
                stroke.ctrls[:, 1] = stroke.ctrls[:, 1]*(y1-y0)/200 + y0
        return strokes

def parse_glyphwiki_tsv(newest_path, all_path):
    data = {}
    for path in [newest_path, all_path]:
        with open(path) as fp:
            for i, line in enumerate(fp):
                if ":" not in line:
                    continue  # header
                if "行" in line:
                    print("ended at line", i)
                    break
                ## parse line
                name, related, text = map(lambda x: x.strip(), line.split('|'))
                name = name.replace("\\", "")
                component_descriptions = text.split('$')
                data[name] = component_descriptions
    return data


def extract_aj1_related(data, cids=[]):
    if len(cids) > 0:
        updated = set(filter(lambda x: x[:4] == "aj1-" and "@" not in x and int(x[4:9]) in cids, data.keys()))
    else:
        updated = set(filter(lambda x: x[:4] == "aj1-" and "@" not in x, data.keys()))
    related = updated.copy()
    print("start tracking aj1-related glyphs...")
    i = 1
    while True:
        print("iteration:", i)
        print(len(related), len(updated))
        i+=1
        next_updated = []
        for name in updated:
            for component_description in data[name]:
                c = component_description.split(':')
                if c[0] == "99":
                    next_updated.append(c[7])
                else:
                    pass
        if len(next_updated) > 0:
            related.update(next_updated)
            updated = next_updated
        else:
            break
    return {k:data[k] for k in related}

def parse_single_line(line, glyph_name=None):
    c = line.split(':')
    # 部品名の確認
    stroke_type = StrokeType(int(c[0]))
    # 各部品クラスの座標確認
    if stroke_type == StrokeType.LINE:  # 直線
        assert(len(c)==7)
        t0, t1, x0, y0, x1, y1 = map(lambda x: int(x), c[1:])
        t1 = t1 if t1 in [313, 413] else t1 % 100  # AJ1範囲には存在しないが、中文対応に必要
        assert(t0 % 100 in [0, 2, 32, 12, 22])
        assert(t1 in [0, 2, 32, 13, 23, 4, 24])
        return KageStroke(stroke_type, StartPointType(t0), EndPointType(t1), [[x0, y0], [x1, y1]])
    elif stroke_type == StrokeType.QCURVE:  # 曲線
        assert(len(c) == 9)
        t0, t1, x0, y0, x1, y1, x2, y2 = map(lambda x: int(x), c[1:])
        if t0 == 2:  # 本来ありえない組み合わせだが稀に存在するので置き換えておく
            t0 = 32
        assert(t0 % 100 in [0, 32, 12, 22, 7, 27])
        assert(t1 % 100 in [7, 0, 8, 4, 5])
        return KageStroke(stroke_type, StartPointType(t0), EndPointType(t1), [[x0, y0], [x1, y1], [x2, y2]])
    elif stroke_type == StrokeType.ORE:  # 折れ
        assert(len(c) == 9)
        t0, t1, x0, y0, x1, y1, x2, y2 = map(lambda x: int(x), c[1:])
        assert(t0 % 100 in [0, 2, 32, 12, 22])  # 直線と同じ
        assert(t1 % 100 in [0, 5, 32])
        return KageStroke(stroke_type, StartPointType(t0), EndPointType(t1), [[x0, y0], [x1, y1], [x2, y2]])
    elif stroke_type == StrokeType.OTSU:  # 乙線
        assert(len(c) == 9)
        t0, t1, x0, y0, x1, y1, x2, y2 = map(lambda x: int(x), c[1:])
        assert(t0 % 100 in [0, 22])
        assert(t1 % 100 in [0, 5])
        return KageStroke(stroke_type, StartPointType(t0), EndPointType(t1), [[x0, y0], [x1, y1], [x2, y2]])
    elif stroke_type == StrokeType.CCURVE:  # 複曲線
        assert(len(c) == 11)
        t0, t1, x0, y0, x1, y1, x2, y2, x3, y3 = map(lambda x: int(x), c[1:])
        assert(t0 % 100 in [0, 32, 12, 22, 7, 27])  # 曲線と同じ
        assert(t1 % 100 in [7, 0, 8, 4, 5])  # 曲線と同じ
        return KageStroke(stroke_type, StartPointType(t0), EndPointType(t1), [[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    elif stroke_type == StrokeType.HARAI:  # 縦払い
        assert(len(c) == 11)
        t0, t1, x0, y0, x1, y1, x2, y2, x3, y3 = map(lambda x: int(x), c[1:])
        assert(t0 % 100 in [0, 2, 32, 12, 22])  # 直線と同じ
        assert(t1 % 100 in [7])
        return KageStroke(stroke_type, StartPointType(t0), EndPointType(t1), [[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
    elif stroke_type == StrokeType.SUBGLYPH:  # 部品引用
        assert(len(c) in [8, 11])
        xd, yd, x0, y0, x1, y1 = map(lambda x: int(x), c[1:7])
        part = c[7]
        if 100 < xd:
            xd -= 200
            if len(c) == 11:
                xs, ys = map(lambda x: int(x), c[9:])
            else:
                # 本来ここは通らない
                # u8f02-jが違反しているので対応
                xs, ys = 0, 0
                print(f"[{glyph_name}] coordinate of S is not found, insert (0, 0): {c}")
        else:
            # kage-engineでは、xd<=100のときSは(0,0)として扱われる
            ## https://github.com/kamichikoichi/kage-engine/blob/97dd6814c2699edbf33b0ee56e8475f6c9b1df1a/kage.js#L101
            # 置換の実態確認のため長さや値のチェックを行っているが結局xs=ys=0を代入しているだけ
            if len(c) == 11:
                xs, ys = map(lambda x: int(x), c[9:])
                if xs == ys == 0:
                    pass
                else:
                    # 本来ここは通らない
                    # u809e-jが違反しているので対応
                    xs, ys = 0, 0
                    print(f"[{glyph_name}] coordinate of S should not be provided, replace it by (0, 0): {c}")
            else:
                xs, ys = 0, 0
        return SubGlyphComponent(part, xd, yd, xs, ys, x0, y0, x1, y1)
    elif stroke_type == StrokeType.SPECIAL:  # 特殊行
        assert(len(c) in [4, 7])  # 仕様上は4または7
        if len(c) == 4:
            assert(c[1] == c[2] == c[3] == "0")  # nop
            return None  # nop instruction - skip this component
        elif len(c) == 7:
            # 図形変形命令はサポートしない
            raise NotImplementedError("transform operation is not supported")

def parse_components(data):
    """
    整形して配列に格納
    """
    result = {}
    for name, component_descriptions in data.items():
        components = []
        for component_description in component_descriptions:
            component = parse_single_line(component_description, glyph_name=name)
            if component is not None:  # Filter out nop instructions
                components.append(component)
        result[name] = components
    return result


def stretch(d, s, x, x0, x1):
    if x < s+100:
        p1 = p3 = x0
        p2 = s+100
        p4 = d+100
    else:
        p1 = s+100
        p3 = d+100
        p2 = p4 = x1
    if p1 == p2:
        return x-p1+p3
    else:
        return (x-p1)/(p2-p1)*(p4-p3) + p3


def expand_glyph(components, data):
    result = []
    for component in components:
        if type(component) is SubGlyphComponent:
            sub_components = copy.deepcopy(expand_glyph(data[component.subglyph_id], data))
            # bbox計算
            bbox = [200, 200, 0, 0] # 必ずアップデートされるような値で初期化
            for c in sub_components:
                b = c.bbox()
                bbox[0] = min(bbox[0], b[0])
                bbox[1] = min(bbox[1], b[1])
                bbox[2] = max(bbox[2], b[2])
                bbox[3] = max(bbox[3], b[3])
            assert(bbox[0]<200 and bbox[1]<200 and 0<bbox[2] and 0<bbox[3])
            # 座標変換
            xd, yd, xs, ys = component.xd, component.yd, component.xs, component.ys
            x0, y0, x1, y1 = component.bbox
            if not (xd == yd == xs == ys == 0):
                # stretchありの場合
                for c in sub_components:
                    for i in range(c.ctrls.shape[0]):
                        x, y = c.ctrls[i]
                        xt = stretch(xd, xs, x, bbox[0], bbox[2])
                        yt = stretch(yd, ys, y, bbox[1], bbox[3])
                        c.ctrls[i, 0] = xt*(x1-x0)/200 + x0
                        c.ctrls[i, 1] = yt*(y1-y0)/200 + y0
            else:
                for c in sub_components:
                    c.ctrls[:, 0] = c.ctrls[:, 0]*(x1-x0)/200 + x0
                    c.ctrls[:, 1] = c.ctrls[:, 1]*(y1-y0)/200 + y0
            result.extend(sub_components)
        else:
            result.append(component)
    return result

from typing import List, Union
import cairosvg
from PIL import Image
import io
from kage_util import KageStroke
import numpy as np

def strokes_to_svg(strokes: List[KageStroke], transform_str: str = None) -> str:
    # KageComponentのリストまたはSVG dコマンド(str)のリストからSVGテキストに変換
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
    svg += f"<g transform='{transform_str}'>" if transform_str is not None else ""
    for c in strokes:
        if type(c) is KageStroke:
            svg += f'<path d="{c.svg_path()}" fill="transparent" stroke="white" stroke-width="3"/>'  # KAGEストロークはアウトラインではなく線なのでfillは透明とする
        elif type(c) is str:
            # ｄコマンドとみなす
            svg += f'<path d="{c}" fill="white" stroke="white"/>'
    svg += "</g>" if transform_str is not None else ""
    svg += '</svg>'
    return svg

def strokes_to_img(strokes: Union[List[KageStroke], List[str]], transform_str: str = None) -> np.ndarray:
    # KageComponentのリストまたはSVG dコマンド(str)のリストから画像に変換
    svg = strokes_to_svg(strokes, transform_str)

    # SVGをPNGのバイトデータに変換
    png_data = cairosvg.svg2png(bytestring=svg.encode('utf-8'))
    # PNGバイトデータをPILイメージに変換
    image = Image.open(io.BytesIO(png_data))
    # ndarrayに変換
    image = np.array(image, dtype=np.uint8)[:, :, 0]
    return image