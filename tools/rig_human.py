import json,struct,sys,numpy as np
from scipy.spatial import cKDTree
def rd(path):
  b=open(path,'rb').read(); L=struct.unpack('<I',b[12:16])[0]; J=json.loads(b[20:20+L]); BIN=b[20+L+8:]
  def acc(i):
    a=J['accessors'][i]; bv=J['bufferViews'][a['bufferView']]; n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    dt={5126:np.float32,5123:np.uint16,5121:np.uint8,5125:np.uint32}[a['componentType']]
    off=bv.get('byteOffset',0)+a.get('byteOffset',0); return np.frombuffer(BIN,dt,a['count']*n,off).reshape(a['count'],n).copy()
  def view(i): bv=J['bufferViews'][i]; o=bv.get('byteOffset',0); return BIN[o:o+bv['byteLength']]
  return J,acc,view
AJ,aacc,_=rd('/home/claude/amirosgame/arin.glb'); prim=AJ['meshes'][0]['primitives'][0]['attributes']
AP=aacc(prim['POSITION']); AJn=aacc(prim['JOINTS_0']).astype(np.int32); AW=aacc(prim['WEIGHTS_0']).astype(np.float32)
if AW.max()>1.5: AW/=255.
IBM=aacc(AJ['skins'][0]['inverseBindMatrices'])
src,out=sys.argv[1],sys.argv[2]
J,acc,view=rd(src); p=J['meshes'][0]['primitives'][0]
idx=acc(p['indices']).reshape(-1).astype(np.uint32); P=acc(p['attributes']['POSITION']); UV=acc(p['attributes']['TEXCOORD_0']); IMGS=[(view(im['bufferView']),im.get('mimeType','image/jpeg')) for im in J['images']]; NRM=acc(p['attributes']['NORMAL']).astype(np.float32)
H=P[:,1].max(); s=1.7/H; Q=P*s                      # uniform into Arin's bind space
# lookup space: stretch x/z so the silhouettes line up, then nearest Arin verts
aw=np.abs(AP[:,0]).max(); qw=np.abs(Q[:,0]).max(); ad=np.abs(AP[:,2]).max(); qd=np.abs(Q[:,2]).max()
L=Q*np.array([aw/qw,1,ad/qd]); tree=cKDTree(AP); d,nn=tree.query(L,k=8)
w=1/(d+1e-4)**2; w/=w.sum(1,keepdims=True)
NJ=36; acc_w=np.zeros((len(Q),NJ),np.float32)
for k in range(8):
  for c in range(4): np.add.at(acc_w,(np.arange(len(Q)),AJn[nn[:,k],c]),w[:,k]*AW[nn[:,k],c])
# keep arms and body apart: arm verts only follow arm bones, body verts never follow hands
jn=[AJ['nodes'][j]['name'] for j in AJ['skins'][0]['joints']]; JP=np.array([np.linalg.inv(IBM[i].reshape(4,4).T)[:3,3] for i in range(NJ)])
par={}; 
for ni,n in enumerate(AJ['nodes']):
  for c in n.get('children',[]): par[c]=ni
jidx={nd:i for i,nd in enumerate(AJ['skins'][0]['joints'])}
segs=[(jidx[par[nd]],i) for nd,i in jidx.items() if nd in par and par[nd] in jidx]
def segd(Pq,a,b):
  A=JP[a]; B=JP[b]; ab=B-A; t=np.clip(((Pq-A)@ab)/max(ab@ab,1e-9),0,1); return np.linalg.norm(Pq-(A+t[:,None]*ab),axis=1)
isarm=np.array([any(k in n for k in ('Arm','Hand','Bone_035','Bone_036')) and 'Shoulder' not in n for n in jn])
dA=np.full(len(L),9.); dB=np.full(len(L),9.); bA=np.zeros(len(L),int)
for a_,b_ in segs:
  dd=segd(L,a_,b_)
  if isarm[b_]: m_=dd<dA; dA[m_]=dd[m_]; bA[m_]=b_
  else: dB=np.minimum(dB,dd)
armv=dA<dB*.7; bodyv=dB<dA*.7
acc_w[np.ix_(armv,~isarm)]=0; acc_w[np.ix_(bodyv,isarm)]=0
z=acc_w.sum(1)<1e-6; acc_w[np.where(z)[0],bA[z]]=1
print('arm verts',armv.sum(),'body',bodyv.sum())
top=np.argsort(-acc_w,1)[:,:4]; tw=np.take_along_axis(acc_w,top,1); tw/=tw.sum(1,keepdims=True)
# normals
N=NRM
# build glb: Arin skeleton nodes + new mesh, no animations
blobs=[]; views=[]; accs=[]
def add(arr,target=None,typ=None,ct=5126,mm=False):
  data=arr.tobytes(); off=sum(len(x) for x in blobs); pad=(-len(data))%4; blobs.append(data+b'\0'*pad)
  v={'buffer':0,'byteOffset':off,'byteLength':len(data)}; 
  if target: v['target']=target
  views.append(v)
  if typ is None: return len(views)-1
  a={'bufferView':len(views)-1,'componentType':ct,'count':len(arr),'type':typ}
  if mm: a['min']=arr.min(0).tolist(); a['max']=arr.max(0).tolist()
  accs.append(a); return len(accs)-1
iI=add(idx,34963,'SCALAR',5125); iP=add(Q.astype(np.float32),34962,'VEC3',mm=True); iN=add(N,34962,'VEC3'); iU=add(UV.astype(np.float32),34962,'VEC2')
iJ=add(top.astype(np.uint16),34962,'VEC4',5123); iW=add(tw.astype(np.float32),34962,'VEC4'); iB=add(IBM.astype(np.float32),None,'MAT4')
vImgs=[add(np.frombuffer(b_,np.uint8)) for b_,_ in IMGS]
G=dict(asset={'version':'2.0'},scene=0,scenes=AJ['scenes'],nodes=AJ['nodes'],skins=[{**AJ['skins'][0],'inverseBindMatrices':iB}],
  meshes=[{'name':'folk','primitives':[{'attributes':{'POSITION':iP,'NORMAL':iN,'TEXCOORD_0':iU,'JOINTS_0':iJ,'WEIGHTS_0':iW},'indices':iI,'material':0}]}],
  materials=J['materials'],textures=J['textures'],samplers=J.get('samplers',[]),images=[{'bufferView':v_,'mimeType':m_} for v_,(_,m_) in zip(vImgs,IMGS)],accessors=accs,bufferViews=views,buffers=[{'byteLength':sum(len(x) for x in blobs)}])
for n in G['nodes']:
  if n.get('mesh') is not None: n['mesh']=0
js=json.dumps(G).encode(); js+=b' '*((-len(js))%4); bn=b''.join(blobs)
o=struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(bn))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(bn),0x004E4942)+bn
open(out,'wb').write(o); print(out,len(o),'H',round(H,3),'verts',len(Q))
