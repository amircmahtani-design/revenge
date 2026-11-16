"""High-quality reduction: quadric decimation that keeps the original UVs, so the original 2048 colour,
normal and roughness maps are reused untouched (no re-bake, no resampling)."""
import json, struct, sys, numpy as np, pymeshlab
src, TARGET, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
b = open(src, 'rb').read(); L = struct.unpack('<I', b[12:16])[0]; J = json.loads(b[20:20+L]); BIN = b[20+L+8:]
def acc(i):
    a = J['accessors'][i]; bv = J['bufferViews'][a['bufferView']]; n = {'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4}[a['type']]
    dt = {5126:np.float32,5123:np.uint16,5125:np.uint32,5121:np.uint8}[a['componentType']]
    return np.frombuffer(BIN, dt, a['count']*n, bv.get('byteOffset',0)+a.get('byteOffset',0)).reshape(a['count'], n).astype(np.float64 if dt==np.float32 else np.int64)
pr = J['meshes'][0]['primitives'][0]; P = acc(pr['attributes']['POSITION']); UV = acc(pr['attributes']['TEXCOORD_0']); F = acc(pr['indices']).reshape(-1, 3)
# weld positions (keep UVs per wedge) so decimation sees one connected surface
q = np.round(P/1e-6).astype(np.int64); _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True); inv = inv.reshape(-1)
Vw = P[first]; Fw = inv[F]; WUV = UV[F].reshape(-1, 2)
import os
stem='/home/claude/work/_tmp_'+os.path.basename(out)
open(stem+'.mtl','w').write('newmtl m0\nmap_Kd dummy.png\n')
with open(stem+'.obj','w') as fo:
    fo.write('mtllib '+os.path.basename(stem)+'.mtl\nusemtl m0\n')
    fo.write(''.join('v %.6f %.6f %.6f\n'%tuple(v) for v in Vw))
    fo.write(''.join('vt %.6f %.6f\n'%tuple(t) for t in UV))
    fo.write(''.join('f %d/%d %d/%d %d/%d\n'%(inv[a]+1,a+1,inv[b]+1,b+1,inv[c]+1,c+1) for a,b,c in F))
ms = pymeshlab.MeshSet(); ms.load_new_mesh(stem+'.obj')
ms.meshing_remove_duplicate_faces(); ms.meshing_remove_null_faces()
ms.meshing_decimation_quadric_edge_collapse_with_texture(targetfacenum=TARGET, qualitythr=.5, extratcoordw=1.0, preserveboundary=True, preservenormal=True, optimalplacement=True, planarquadric=True)
cm = ms.current_mesh(); V2 = cm.vertex_matrix(); F2 = cm.face_matrix(); W2 = cm.wedge_tex_coord_matrix().reshape(-1, 3, 2)
# split vertices by (position, uv)
key = np.concatenate([np.repeat(F2.reshape(-1, 1), 1, 1).astype(np.float64), np.round(W2.reshape(-1, 2)*1e5)], 1)
_, fi, inv2 = np.unique(key, axis=0, return_index=True, return_inverse=True); inv2 = inv2.reshape(-1)
Pn = V2[F2.reshape(-1)[fi]]; UVn = W2.reshape(-1, 2)[fi]; Fn = inv2.reshape(-1, 3)
# smooth normals from welded topology
fn = np.cross(V2[F2[:,1]]-V2[F2[:,0]], V2[F2[:,2]]-V2[F2[:,0]]); Nv = np.zeros_like(V2)
for c in range(3): np.add.at(Nv, F2[:, c], fn)
Nv /= np.linalg.norm(Nv, axis=1, keepdims=True)+1e-12; Nn = Nv[F2.reshape(-1)[fi]]
lo = Pn.min(0); hi = Pn.max(0); Pn = Pn-[(lo[0]+hi[0])/2, lo[1], (lo[2]+hi[2])/2]
print('tris', len(Fn), 'verts', len(Pn), 'dims', (hi-lo).round(3))
blobs = []; views = []; accs = []
def add(arr, target=None, typ=None, ct=5126, mm=False):
    data = arr.tobytes(); off = sum(len(x) for x in blobs); blobs.append(data+b'\0'*((-len(data)) % 4))
    v = {'buffer':0, 'byteOffset':off, 'byteLength':len(data)}
    if target: v['target'] = target
    views.append(v)
    if typ is None: return len(views)-1
    a = {'bufferView':len(views)-1, 'componentType':ct, 'count':len(arr), 'type':typ}
    if mm: a['min'] = arr.min(0).tolist(); a['max'] = arr.max(0).tolist()
    accs.append(a); return len(accs)-1
iI = add(Fn.astype(np.uint32).reshape(-1), 34963, 'SCALAR', 5125); iP = add(Pn.astype(np.float32), 34962, 'VEC3', mm=True); iN = add(Nn.astype(np.float32), 34962, 'VEC3'); iU = add(UVn.astype(np.float32), 34962, 'VEC2')
imgs = []
for im in J['images']:
    bv = J['bufferViews'][im['bufferView']]; o = bv.get('byteOffset', 0); imgs.append({'bufferView':add(np.frombuffer(BIN[o:o+bv['byteLength']], np.uint8)), 'mimeType':im.get('mimeType', 'image/jpeg')})
G = dict(asset={'version':'2.0'}, scene=0, scenes=[{'nodes':[0]}], nodes=[{'name':out, 'mesh':0}],
  meshes=[{'primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU}, 'indices':iI, 'material':0}]}],
  materials=J['materials'], textures=J['textures'], samplers=J.get('samplers', []), images=imgs, accessors=accs, bufferViews=views, buffers=[{'byteLength':sum(len(x) for x in blobs)}])
js = json.dumps(G).encode(); js += b' '*((-len(js)) % 4); bn = b''.join(blobs)
open(out, 'wb').write(struct.pack('<III', 0x46546C67, 2, 28+len(js)+len(bn))+struct.pack('<II', len(js), 0x4E4F534A)+js+struct.pack('<II', len(bn), 0x004E4942)+bn)
