"""Rotate a T-pose character's arms down into Arin's A-pose (about 67 degrees below horizontal) so the
Amiros human skeleton and its animations fit. Rewrites only POSITION/NORMAL in place in a copy of the GLB."""
import json, struct, sys, numpy as np
src, out = sys.argv[1], sys.argv[2]; THETA = np.radians(float(sys.argv[3]) if len(sys.argv) > 3 else 67)
b = bytearray(open(src, 'rb').read()); L = struct.unpack('<I', b[12:16])[0]; J = json.loads(b[20:20+L]); B0 = 20+L+8
def arr(i):
    a = J['accessors'][i]; bv = J['bufferViews'][a['bufferView']]; off = B0+bv.get('byteOffset', 0)+a.get('byteOffset', 0)
    return off, np.frombuffer(bytes(b[off:off+a['count']*12]), np.float32).reshape(-1, 3).copy()
pr = J['meshes'][0]['primitives'][0]['attributes']
offP, P = arr(pr['POSITION']); offN, N = (arr(pr['NORMAL']) if 'NORMAL' in pr else (None, None))
x, y = P[:, 0], P[:, 1]
band = (np.abs(x) > .38) & (np.abs(x) < .5)
ya = float(np.median(y[band])); th = float(np.percentile(y[band], 95) - np.percentile(y[band], 5))
side = (np.abs(y-(ya-.13)) < .02) & (np.abs(x) < .34); px = float(np.abs(x[side]).max()) + .01
pivot_y = ya
print('arm centre y', round(ya, 3), 'thickness', round(th, 3), 'pivot x', round(px, 3))
ax = np.abs(x)
w = np.clip((ax-(px-.06))/.13, 0, 1); w = w*w*(3-2*w)
w[y < ya-th*.5-.06] = 0                      # torso below the armpit stays put
w[(ax < px-.02) & (y > ya+th*.5)] = 0        # neck / head / collar
for s in (1, -1):
    m = (np.sign(x) == s) & (w > 0)
    a = -s*THETA*w[m]; c, sn = np.cos(a), np.sin(a)
    dx, dy = P[m, 0]-s*px, P[m, 1]-pivot_y
    P[m, 0] = s*px+dx*c-dy*sn; P[m, 1] = pivot_y+dx*sn+dy*c
    if N is not None:
        nx, ny = N[m, 0].copy(), N[m, 1].copy(); N[m, 0] = nx*c-ny*sn; N[m, 1] = nx*sn+ny*c
b[offP:offP+P.nbytes] = P.astype(np.float32).tobytes()
if N is not None: b[offN:offN+N.nbytes] = N.astype(np.float32).tobytes()
a = J['accessors'][pr['POSITION']]; a['min'] = P.min(0).tolist(); a['max'] = P.max(0).tolist()
js = json.dumps(J, separators=(',', ':')).encode(); js += b' '*((-len(js)) % 4)
rest = bytes(b[20+L:]); o = struct.pack('<III', 0x46546C67, 2, 12+8+len(js)+len(rest))+struct.pack('<II', len(js), 0x4E4F534A)+js+rest
open(out, 'wb').write(o); print(out, 'dims', (P.max(0)-P.min(0)).round(3))
