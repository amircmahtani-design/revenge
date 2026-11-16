import json, struct, sys, numpy as np
src, out, S = sys.argv[1], sys.argv[2], float(sys.argv[3])

def rd(path):
    b = open(path, 'rb').read(); L = struct.unpack('<I', b[12:16])[0]; J = json.loads(b[20:20+L]); BIN = b[20+L+8:]
    def acc(i):
        a = J['accessors'][i]; bv = J['bufferViews'][a['bufferView']]; n = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
        dt = {5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']]
        off = bv.get('byteOffset',0)+a.get('byteOffset',0); return np.frombuffer(BIN, dt, a['count']*n, off).reshape(a['count'], n).copy()
    def view(i): bv = J['bufferViews'][i]; o = bv.get('byteOffset',0); return BIN[o:o+bv['byteLength']]
    return J, acc, view

J, acc, view = rd(src); p = J['meshes'][0]['primitives'][0]
idx = acc(p['indices']).reshape(-1).astype(np.uint32); P = acc(p['attributes']['POSITION']).astype(np.float64); UV = acc(p['attributes']['TEXCOORD_0'])
IMGS = [(view(im['bufferView']), im.get('mimeType','image/jpeg')) for im in J['images']]

# joints in the model's own space (z forward, y up, feet at y=0)
X = .12
JD = [  # name, parent, position
 ('hips',  None,   (0, .78, -.45)),
 ('spine', 'hips', (0, .80, -.12)),
 ('chest', 'spine',(0, .82,  .24)),
 ('neck',  'chest',(0, .86,  .48)),
 ('head',  'neck', (0, .98,  .68)),
]
for s, nm in ((1,'L'),(-1,'R')):
    JD += [('shoulder'+nm,'chest',(s*X,.74,.42)),('elbow'+nm,'shoulder'+nm,(s*X,.40,.36)),('wrist'+nm,'elbow'+nm,(s*X,.12,.40))]
    JD += [('thigh'+nm,'hips',(s*X,.72,-.48)),('knee'+nm,'thigh'+nm,(s*X,.45,-.36)),('hock'+nm,'knee'+nm,(s*X,.20,-.55))]
TAIL = [(0,.84,-.70),(0,.66,-.80),(0,.46,-.83),(0,.28,-.85),(0,.13,-.89),(0,.08,-.96)]
for i, t in enumerate(TAIL): JD.append(('tail%d' % i, 'hips' if i == 0 else 'tail%d' % (i-1), t))
names = [j[0] for j in JD]; ji = {n:i for i, n in enumerate(names)}; JP = np.array([j[2] for j in JD], float)

# bone segments used for weighting: (joint that owns it, start, end)
ENDS = {'head':(0,.96,.95), 'neck':None}
segs = []
for i, (n, par, pos) in enumerate(JD):
    kids = [k for k, j in enumerate(JD) if j[1] == n]
    for k in kids: segs.append((i, JP[i], JP[k]))
for s, nm in ((1,'L'),(-1,'R')):
    segs.append((ji['wrist'+nm], JP[ji['wrist'+nm]], np.array((s*X,.03,.50))))
    segs.append((ji['hock'+nm], JP[ji['hock'+nm]], np.array((s*X,.03,-.45))))
segs.append((ji['head'], JP[ji['head']], np.array((0,.96,.95))))
segs.append((ji['tail5'], JP[ji['tail5']], np.array((0,.08,-.99))))

def segd(Q, A, B):
    ab = B-A; t = np.clip(((Q-A)@ab)/max(ab@ab,1e-9), 0, 1); return np.linalg.norm(Q-(A+t[:,None]*ab), axis=1)

NJ = len(JD); D = np.full((len(P), NJ), 9.0)
for j, A, B in segs: D[:, j] = np.minimum(D[:, j], segd(P, A, B))
x, y, z = P[:,0], P[:,1], P[:,2]
isleg = lambda n: any(n.startswith(k) for k in ('shoulder','elbow','wrist','thigh','knee','hock'))
for j, n in enumerate(names):
    if isleg(n):
        side = 1 if n.endswith('L') else -1; front = n.startswith(('shoulder','elbow','wrist'))
        bad = (x*side < -.015) | ((z < -.05) if front else (z > .0))
        D[bad, j] = 9
    if n.startswith('tail'):
        D[(np.abs(x) > .085) | (z > -.6), j] = 9
# legs never pull from the head/neck; tail only near the rump
for j in (ji['neck'], ji['head']): D[y < .55, j] = 9
W = 1.0/np.maximum(D, .012)**4
dmin = D.min(1, keepdims=True); W[D > dmin*1.6 + .02] = 0
top = np.argsort(-W, 1)[:, :4]; tw = np.take_along_axis(W, top, 1); tw /= tw.sum(1, keepdims=True)
print('joints', NJ, 'verts', len(P), 'unweighted', int((W.sum(1) == 0).sum()))

# smooth normals on welded positions
q = np.round(P/1e-5).astype(np.int64); _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True); inv = inv.reshape(-1)
F = idx.reshape(-1, 3); fn = np.cross(P[F[:,1]]-P[F[:,0]], P[F[:,2]]-P[F[:,0]])
Nw = np.zeros((len(first), 3));
for c in range(3): np.add.at(Nw, inv[F[:,c]], fn)
Nw /= np.linalg.norm(Nw, axis=1, keepdims=True)+1e-12; N = Nw[inv].astype(np.float32)

Q = (P*S).astype(np.float32); JPs = JP*S
blobs = []; views = []; accs = []
def add(arr, target=None, typ=None, ct=5126, mm=False):
    data = arr.tobytes(); off = sum(len(b) for b in blobs); blobs.append(data+b'\0'*((-len(data)) % 4))
    v = {'buffer':0, 'byteOffset':off, 'byteLength':len(data)}
    if target: v['target'] = target
    views.append(v)
    if typ is None: return len(views)-1
    a = {'bufferView':len(views)-1, 'componentType':ct, 'count':len(arr), 'type':typ}
    if mm: a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
    accs.append(a); return len(accs)-1
IBM = np.zeros((NJ, 16), np.float32)
for i in range(NJ):
    m = np.eye(4); m[:3, 3] = -JPs[i]; IBM[i] = m.T.reshape(-1)
iI = add(idx, 34963, 'SCALAR', 5125); iP = add(Q, 34962, 'VEC3', mm=True); iN = add(N, 34962, 'VEC3'); iU = add(UV.astype(np.float32), 34962, 'VEC2')
iJ = add(top.astype(np.uint16), 34962, 'VEC4', 5123); iW = add(tw.astype(np.float32), 34962, 'VEC4'); iB = add(IBM, None, 'MAT4'); vImgs = [add(np.frombuffer(b_, np.uint8)) for b_, _ in IMGS]
nodes = []
for i, (n, par, pos) in enumerate(JD):
    lp = JPs[i] - (JPs[ji[par]] if par else 0)
    nodes.append({'name':n, 'translation':[float(v) for v in lp], 'children':[k for k, j in enumerate(JD) if j[1] == n]})
for nd in nodes:
    if not nd['children']: del nd['children']
meshNode = len(nodes); nodes.append({'name':'panther', 'mesh':0, 'skin':0})
rootNode = len(nodes); nodes.append({'name':'Panther', 'children':[0, meshNode]})
G = dict(asset={'version':'2.0'}, scene=0, scenes=[{'nodes':[rootNode]}], nodes=nodes, skins=[{'joints':list(range(NJ)), 'inverseBindMatrices':iB, 'skeleton':0}],
  meshes=[{'name':'panther', 'primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW}, 'indices':iI, 'material':0}]}],
  materials=J['materials'], textures=J['textures'], samplers=J.get('samplers', []), images=[{'bufferView':v_, 'mimeType':m_} for v_, (_, m_) in zip(vImgs, IMGS)], accessors=accs, bufferViews=views, buffers=[{'byteLength':sum(len(b) for b in blobs)}])
js = json.dumps(G).encode(); js += b' '*((-len(js)) % 4); bn = b''.join(blobs)
o = struct.pack('<III', 0x46546C67, 2, 12+8+len(js)+8+len(bn))+struct.pack('<II', len(js), 0x4E4F534A)+js+struct.pack('<II', len(bn), 0x004E4942)+bn
open(out, 'wb').write(o); print(out, len(o), 'height', round(Q[:,1].max(), 3))
# debug: dominant joint per vertex colour map (side view)
if len(sys.argv) > 4:
    from PIL import Image, ImageDraw
    rng = np.random.RandomState(3); cols = (rng.rand(NJ, 3)*200+55).astype(int)
    im = Image.new('RGB', (900, 600), (25,25,25)); d = ImageDraw.Draw(im); dom = top[:,0]
    o_ = np.argsort(x)
    for k in o_: d.point(((z[k]+1)*450, (1.25-y[k])*450), fill=tuple(cols[dom[k]]))
    for i in range(NJ): d.text(((JP[i,2]+1)*450, (1.25-JP[i,1])*450), names[i][:6], fill=(255,255,255))
    im.save(sys.argv[4])
