# G2G_Sparse_SNN

# Installation

```
pip install snntorch torchvision
```

**NVIDIA (CUDA):**

```
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**AMD GPU:**

```
pip install torch-directml
```
# Run
Run the script using `python train.py --model X`, where **X** can be one of `dense`, `index`, `random`, or `mixer`.

## Arguments
| Argument    | Description                         | Default |
| ----------- | ----------------------------------- | ------- |
| `--model`   | `dense`, `index`, `random`, `mixer` | –       |
| `--p_inter` | Inter-group probability (p′)        | 0.15    |
| `--epochs`  | Number of training epochs           | 20      |

## Example commands:
### Index-based sparse SNN (default settings)
`python evaluation/train.py --model index`

### Mixer-style SNN with inter-group probability p'=0.20
`python evaluation/train.py --model mixer --p_inter 0.20`

### Random sparse SNN with custom epochs and p'
`python evaluation/train.py --model random --epochs 30 --p_inter 0.10`
