# Selective vision–language classification of post-flood facade defects

Code for the experiments of:

> *A post-flood building defect dataset and selective vision–language classification with expert deferral for damage assessment.*
> Oriol Chacón-Albero, Mariana Castañeda Jaramillo, Raimon Calabuig-Moreno, Jaume Jordán, Vicente Julian, Vicent Botti. 2026 < once published>

The data are the **RIUADA** dataset, released separately on Zenodo:
<https://doi.org/10.5281/zenodo.23536529>

The code evaluates few-shot adaptation of frozen CLIP-family models (zero-shot,
visual prototypes, combined text–visual prototypes, Tip-Adapter, Tip-Adapter-F)
on the RIUADA lesion crops, and analyses selective prediction: how much of the
workload a model can classify reliably, and which samples it should defer to an
expert.

## 1. Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA driver
pip install -r requirements.txt
```

A GPU is needed for `run_ladder.py` (feature extraction with ViT-H/14).
`analyze_selective.py` runs on CPU. The OpenCLIP weights (about 4 GB per
backbone) are downloaded automatically on first use.

## 2. Data

Download `crops.zip` from the Zenodo record and unzip it in the repository root:

```
crops/
├── Corrosion/
├── Cracks/
├── Detachment/
├── Spalling/
└── Stains/
```

Only the crops are needed; the photographs, annotations and attribute table on
Zenodo are not used by this code.

## 3. Running

**Step 1: adaptation experiments** (GPU)

```bash
python run_ladder.py
```

Settings are the constants at the top of the file (backbones, annotation
budgets `K_LIST`, number of runs, seeds, Tip-Adapter hyperparameters). The
committed values are those of the paper. For each backbone, the script

1. extracts and caches image features (`results/<backbone>/features.pt`,
   `features_aug.pt`); this is the slow step and is done once;
2. for each budget K and each of the 30 runs, samples K support crops per class,
   evaluates all methods on the remaining crops and saves all logits to
   `results/<backbone>/k<K>/runs/run_XX.npz`;
3. writes mean and standard deviation of accuracy and macro-F1 to
   `results/summary.json`.

**Step 2: selective prediction** (CPU)

```bash
python analyze_selective.py results/metaclip_fullcc/k16 --method tip_f
python analyze_selective.py results/laion2b_s32b_b79k/k16 --method combproto
python analyze_selective.py results/metaclip_fullcc/k16 --method combproto
python analyze_selective.py results/laion2b_s32b_b79k/k16 --method tip_f
```

Outputs go to `results/<backbone>/k<K>/analysis_<method>/`: risk–coverage and
confusion-matrix figures, and `selective_summary.json` with every reported
number (selective accuracy at fixed coverage, coverage at fixed accuracy,
AURC/E-AURC, the paired branch-consensus vs. MSP comparison with its Wilcoxon
test, per-class deferral).

## 4. Reproducibility notes

- **Splits** depend only on the seed and the sorted file list of each class
  folder, so they are identical across methods, scores and backbones, and the
  support set at a budget K is contained in the support set at every larger
  budget.
- **Class order matters.** Support sets are drawn class by class from one
  random stream, in the order of `CLASSES` in `run_ladder.py`. Keep that order
  to reproduce the paper's splits.
- **Feature caches** are reused if the list of file paths is unchanged. If you
  replace the images under the same paths, delete the `features*.pt` files.
- **No tuning on test data.** Tip-Adapter's α = 1 and β = 5.5 are the original
  ImageNet values, fixed a priori.
- **Released vs. original crops.** The paper's results were computed on the
  crops before privacy masking. Running this code on the released crops
  reproduces every reported metric within ±0.15 percentage points.

## 5. Citation

```bibtex
@article{chacon2026selective,
  title   = {A post-flood building defect dataset and selective vision--language
             classification with expert deferral for damage assessment},
  author  = {Chacón-Albero, Oriol and
                  Castañeda Jaramillo, Mariana and
                  Calabuig-Moreno, Raimon and
                  Jordán, Jaume and
                  Julian, Vicente and
                  Botti, Vicent},
  year    = {2026}
}

@dataset{riuada_dataset,
  author       = {Chacón-Albero, Oriol and
                  Castañeda Jaramillo, Mariana and
                  Calabuig-Moreno, Raimon and
                  Jordán, Jaume and
                  Julian, Vicente and
                  Botti, Vicent},
  title        = {{RIUADA}: an expert-annotated dataset of facade
                   defects in flood-affected residential buildings
                   after the 2024 {V}alencia floods
                  },
  note   = {[dataset]},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.23536529}
}
```

## 6. Licence

Code: MIT (see `LICENSE`). Data: CC BY-NC 4.0, with a data protection notice;
see the Zenodo record.