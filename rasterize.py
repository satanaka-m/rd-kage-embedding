import numpy as np
import freetype
from fontTools.ttLib import TTFont
from fontTools.misc.transform import Transform
import cv2
import time

class CidFace(freetype.Face):
    def __init__(self, font_path):
        super().__init__(font_path)
        reverse_glyph_map = TTFont(font_path).getReverseGlyphMap()  # cid -> glyph_id
        self.cid2gid = {int(cid[3:]): gid for cid, gid in reverse_glyph_map.items() if cid.startswith("cid")}

    def load_glyph(self, cid, flags=4):
        super().load_glyph(self.cid2gid.get(cid), flags)

    def get_advance(self, cid, flags):
        return super().get_advance(self.cid2gid.get(cid), flags)
      
def rasterize(face: CidFace, cid: int, width: int, height: int, transform: Transform=None) -> np.ndarray:
    # グリフのメトリクスを取得し、位置合わせとTransformの適用を同時に行う
    u = face.units_per_EM
    face.set_pixel_sizes(u, u)
    advance = face.get_advance(cid, freetype.FT_LOAD_NO_SCALE)
    if transform is None:
        transform = Transform()
    t = transform.translate(width/2, 0).scale(height/u).translate(-advance/2, -face.descender)
    m = np.array([[float(t.xx), float(t.yx)], [float(t.xy), float(t.yy)]], dtype=np.float32)
    d = np.array([float(t.dx), float(t.dy)], dtype=np.float32)
    m_fixed = freetype.FT_Matrix.from_buffer_copy((m*2**16).astype(np.int64).tobytes())  # to 16.16 fixed format
    d_fixed = freetype.FT_Vector.from_buffer_copy((d*2**6).astype(np.int64).tobytes())  # to 26.6 fixed format
    face.set_transform(m_fixed, d_fixed)

    # グリフ読み込みとレンダリング
    face.load_glyph(cid)
    glyph = face.glyph
    glyph.render(freetype.FT_RENDER_MODE_NORMAL)

    # 指定したサイズのndarrayにコピー
    bitmap = glyph.bitmap
    glyph_width, glyph_height = bitmap.width, bitmap.rows

    try:
        img = np.ctypeslib.as_array(bitmap._FT_Bitmap.buffer, (glyph_height, glyph_width))
    except ValueError:
        img = np.zeros((glyph_height, glyph_width))
    # 左右へのはみ出し対応
    x0 = glyph.bitmap_left
    if x0 < 0:
        x0, x1 = 0, -x0
    else:
        x0, x1 = x0, 0
    # 上下へのはみ出し対応
    y0 = height - glyph.bitmap_top
    if y0 < 0:
        y0, y1 = 0, -y0
    else:
        y0, y1 = y0, 0
    # imgとoutputの共通領域のサイズ
    w = min(glyph_width, width-x0, glyph_width-x1)
    h = min(glyph_height, height-y0, glyph_height-y1)

    output = np.zeros((height, width), dtype=np.uint8)
    if w <= 0 or h <= 0:  # imgとoutputで共通領域がない場合
        return output
    else:
        output[y0:y0+h, x0:x0+w] = img[y1:y1+h, x1:x1+w]
        return output


if __name__ == "__main__":
    
    # font_path = "./data/fonts/AP-OTF-UDShinGoCOsezPr6N-Reg.otf"
    font_path = "data/fonts/AP-OTF-TakaHandStdN-Light.otf"
    cid = 1260
    width = height = 128

    face = CidFace(font_path)

    # transformの確認
    for tx, ty in [[0, 0], [-20, 0], [20, 0], [0, -20], [0, 20]]:
        print(tx, ty)
        transform = Transform().translate(width/2, height/2).scale(-1, 1.1).translate(-width/2, -height/2)
        img = rasterize(face, cid, width, height, Transform().translate(tx, ty).transform(transform))
        cv2.imshow("img", img)
        cv2.waitKey(0)
        del img

    # 実行時間計測
    n = 1000
    start_time = time.time()
    for i in range(n):
        img = rasterize(face, cid, width, height)
    end_time = time.time()
    execution_time_per_image = (end_time - start_time)/n
    print("Execution time:", execution_time_per_image*1000000, "us")

    # 描画したビットマップを確認
    cv2.imshow("img", img)
    cv2.waitKey(0)
