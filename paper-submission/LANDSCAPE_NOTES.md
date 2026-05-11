# Robot Learning Perception Landscape — Research Notes

## The Unjustified Backbone Choice Problem

Every major robot learning system makes an implicit perception backbone choice
that is never systematically justified:

### VLAs (Vision-Language-Action Models)
| System | Visual backbone | Justified? |
|--------|----------------|------------|
| π₀ (Physical Intelligence) | SigLIP (from PaliGemma 3B) | No — architecture choice |
| OpenVLA | DINOv2 + SigLIP fused | Partially — "fuses spatial + semantic" |
| RT-2 | PaLI ViT encoder | No — inherited from VLM |
| Octo | ViT (from VC-1 or scratch) | No |
| GR00T N1/N1.6 (NVIDIA) | Eagle-2 vision encoder | No |
| SmolVLA | SigLIP (compact) | Size-driven, not quality-driven |
| GR-2 | Video encoder | No |
| RoboVLMs (Nature MI, 2025) | **Studied**: DINOv2, SigLIP, CLIP variants | **YES — this is the closest work to RPX's thesis** |

### Modular Manipulation Systems  
| System | Perception modules | Justified? |
|--------|-------------------|------------|
| OK-Robot | GroundingDINO + Lang-SAM + AnyGrasp (depth) | One ablation, not systematic |
| 3D Diffusion Policy (DP3) | ZoeDepth → point cloud | No — "we use ZoeDepth" |
| iTeach / HRT1 | UOIS segmentation | No |
| SceneReplica | Swappable perception → grasp planning | Showed perception is bottleneck |

### World Models
| System | Perception consumed | Justified? |
|--------|-------------------|------------|
| TesserAct (ICCV 2025) | Generates RGB + Depth + Normal videos | Needs good depth GT for training |
| GWM (ICCV 2025) | 3D Gaussian-based world model | Consumes RGB-D for scene state |
| RoboScape (NeurIPS 2025) | Physics-informed video prediction | Needs perception for state estimation |
| MVISTA-4D | 4D scene dynamics prediction | Consumes posed RGB |
| DexWM | Latent state from perception | Perception quality = state quality |

## Key Finding: RoboVLMs (Nature Machine Intelligence, 2025)

Li et al. "What matters in building vision-language-action models for 
generalist robots" — Nature Machine Intelligence, Feb 2026.

This paper asks THREE design questions for VLAs:
1. **Which backbone to select** (DINOv2 vs SigLIP vs CLIP vs combinations)
2. How to formulate VLA architectures 
3. When to add cross-embodiment data

Key finding: **backbone choice matters enormously** — but they evaluate
on SimplerEnv (simulation) and CALVIN (simulation), not real-world
deployment conditions.

**RPX's positioning against RoboVLMs:**
- RoboVLMs asks "which backbone?" in simulation → RPX asks it in real world
- RoboVLMs evaluates end-to-end policy success → RPX evaluates the perception
  signals that these backbones produce BEFORE policy consumption
- RoboVLMs can't decompose: is the failure perception or policy? → RPX isolates
  perception quality

## The In-Context Prompting Angle

RPX has single-object scenes (with AR tags) + multi-object scenes (3-phase).
Same objects in both. This enables:

### In-Context Visual Prompting for Detection/Grounding
- Give the model a clean, isolated image of the object (from single-obj scene)
- Ask it to find/segment that object in the cluttered multi-object scene
- This is exactly what DE-ViT (CoRL 2024) and DetPO do — few-shot detection
  with visual exemplars

### Why this is powerful for the paper:
1. Tests in-context learning capability of VLMs on real robot-relevant objects
2. Bridges the "novel instance" problem — the object IS novel to the model
3. The single-object image is what a robot would capture during "object registration"
   — a standard step in real deployment
4. Compares: text-prompt grounding vs visual-prompt grounding on same objects
5. The 3-phase structure tests: does visual prompting degrade during interaction?

### Concrete task formulation:
- **Input**: reference image (from single-obj scene) + query image (from multi-obj scene)
- **Output**: bounding box / mask of the target object in the query image
- **GT**: masks from multi-obj scene annotation
- **Models**: DE-ViT, NIDS-Net, CogVLM2, Qwen2-VL (with visual prompt)
- **Evaluation**: same scene in clutter vs interaction vs clean

This is NOT a new task — it's a new evaluation AXIS for detection/grounding.
