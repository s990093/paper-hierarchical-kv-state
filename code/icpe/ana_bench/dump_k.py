"""用 p3_fvar 的 V1 修補算出 x1_sim 的常數 K，存成 JSON（給沒有 pandas 的直譯器用）。"""
import sys, os, json
code, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(code, "p3x")); sys.path.insert(0, code); sys.path.insert(0, os.path.join(code, "icpe"))
import p3_fvar
note = {}
XS = p3_fvar.patch_x1("V1", os.path.join(os.path.dirname(code), "results/icpe_h200/p01/calib"), os.path.join(sys.argv[3]), note)
K = XS.constants()
s = json.dumps(K); assert json.loads(s) == K
open(out, "w").write(s); print("K keys", sorted(K), "f_ms len", len(K["f_ms"]), "note", note)
