# Revenge

A browser game (Three.js r128) set in the Kumaon hills in 1926. You play the leopard; the hunters waiting in their machans are the prey.

- `index.html`: the whole game (terrain, forest, village, rope bridge, machans, panther controls, hunters, dusk and night)
- `models/panther.glb`: the Meshy panther on a 23-bone quadruped rig
- `models/hunter.glb`, `models/safari.glb`: the Meshy hunters, rigged on the Amiros human skeleton
- `models/human_anims.glb`: shared human clips (idle, walk, run, crouch, climb)
- `tex/`: ground, rock and slate textures; `img/title.jpg`: title art
- `lib/`: Three.js r128, GLTFLoader, SkeletonUtils
- `tools/`: the scripts used to reduce, repose and rig the Meshy models

Controls: WASD move, Shift run, C stalk, Space pounce (under a machan it leaps up), N dusk/night, M sound.
