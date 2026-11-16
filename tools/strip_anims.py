"""Keep only the skeleton nodes and animations of a GLB (drop meshes, textures, skins)."""
import json, struct, sys
src, out = sys.argv[1], sys.argv[2]; keep_names = sys.argv[3].split(',') if len(sys.argv) > 3 else None
b = open(src, 'rb').read(); L = struct.unpack('<I', b[12:16])[0]; J = json.loads(b[20:20+L]); BIN = b[20+L+8:]
anims = [a for a in J['animations'] if keep_names is None or a.get('name') in keep_names]
for a in J['animations']:
    if a in anims: continue
used = []; amap = {}
def remap(i):
    if i not in amap: amap[i] = len(used); used.append(i)
    return amap[i]
for a in anims:
    for s in a['samplers']: s['input'] = remap(s['input']); s['output'] = remap(s['output'])
blob = bytearray(); views = []; accs = []
for i in used:
    a = dict(J['accessors'][i]); bv = J['bufferViews'][a['bufferView']]
    o = bv.get('byteOffset', 0); data = BIN[o:o+bv['byteLength']]
    while len(blob) % 4: blob.append(0)
    views.append({'buffer':0, 'byteOffset':len(blob), 'byteLength':len(data)}); blob += data; a['bufferView'] = len(views)-1; accs.append(a)
nodes = []
for n in J['nodes']:
    m = {k:v for k, v in n.items() if k not in ('mesh', 'skin')}; nodes.append(m)
G = {'asset':{'version':'2.0'}, 'scene':0, 'scenes':J['scenes'], 'nodes':nodes, 'animations':anims, 'accessors':accs, 'bufferViews':views, 'buffers':[{'byteLength':len(blob)}]}
js = json.dumps(G, separators=(',', ':')).encode(); js += b' '*((-len(js)) % 4)
while len(blob) % 4: blob.append(0)
o = struct.pack('<III', 0x46546C67, 2, 28+len(js)+len(blob))+struct.pack('<II', len(js), 0x4E4F534A)+js+struct.pack('<II', len(blob), 0x004E4942)+bytes(blob)
open(out, 'wb').write(o); print(out, len(o), [a.get('name') for a in anims])
