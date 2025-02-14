# minLoRAplus

A fork of the excellent [minLoRA](https://github.com/cccntu/minLoRA) repo by [cccntu](https://github.com/cccntu), with functionality added for Passio use cases

A minimal, but versatile PyTorch re-implementation of [LoRA](https://github.com/microsoft/LoRA). 

In only ~100 lines of code, minLoRA supports the following features:

- Functional, no need to modify the model definition
- Works everywhere, as long as you use `torch.nn.Module`
- PyTorch native, uses PyTorch's `torch.nn.utils.parametrize` to do all the heavy lifting
- Easily extendable, you can add your own LoRA parameterization
- Supports training, inference, and inference with multiple LoRA models

### Plus:
- Finetune any [timm](https://github.com/huggingface/pytorch-image-models) models using LoRA:
  - `ViTClassifier` module to train any `timm` model with transformer architecture
  - `CNNClassifier` module to train any `timm` CNN.


## Basic Usage

- `demo.ipynb` shows the basic usage of the library
- `advanced_usage.ipynb` shows how you can add LoRA to other layers such as embedding, and how to tie weights

## Examples

- Finetuning GPT using LoRA + nanoGPT: https://github.com/cccntu/LoRAnanoGPT/pull/1/files
- Sample training code for `timm` ViT: `vit_trainer.ipynb`
- Sample training code for `timm` CNN: `cnn_trainer.ipynb`

## Running Trainers

If you just want to run the notebooks make sure the requirements are installed first:

```
pip install -r requirements.txt
```

Then run `jupyter notebook`.

## Library Installation

If you want to `import minloraplus` into your project:

```
git clone https://github.com/Passiolife/minLoRAplus.git
cd minLoRAplus
pip install -e .
```

## Usage

```python
import torch
from minloraplus import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora,
    remove_lora, load_multiple_lora, select_lora
```

### Training a model with minLoRAplus

```python
model = torch.nn.Linear(in_features=5, out_features=3)
# Step 1: Add LoRA to the model
add_lora(model)

# Step 2: Collect the parameters, pass them to the optimizer

parameters = [
    {"params": list(get_lora_params(model))},
]
optimizer = torch.optim.AdamW(parameters, lr=1e-3)

# Step 3: Train the model
# ...

# Step 4: export the LoRA parameters
lora_state_dict = get_lora_state_dict(model)
```

### Loading and Inferencing with minLoRAplus

```python
# Step 1: Add LoRA to your model
add_lora(model)

# Step 2: Load the LoRA parameters
_ = model.load_state_dict(lora_state_dict, strict=False)

# Step 3: Merge the LoRA parameters into the model
merge_lora(model)
```

### Inferencing with multiple LoRA models

```python
# to avoid re-adding lora to the model when rerun the cell, remove lora first
remove_lora(model)
# Step 1: Add LoRA to your model
add_lora(model)

# Step 2: Load the LoRA parameters

# load three sets of LoRA parameters
lora_state_dicts = [lora_state_dict_0, lora_state_dict_1, lora_state_dict_2]

load_multiple_lora(model, lora_state_dicts)


# Step 3: Select which LoRA to use at inference time
Y0 = select_lora(model, 0)(x)
Y1 = select_lora(model, 1)(x)
Y2 = select_lora(model, 2)(x)
```
### References

- [microsoft/LoRA](https://github.com/microsoft/LoRA) has the official implementation of LoRA, in PyTorch
- [karpathy/minGPT](https://github.com/karpathy/minGPT) the structure of the repo is adapted from minGPT
- [cccntu/minLoRA](https://github.com/cccntu/minLoRA) the original repo minLoRAplus is based on

### TODO
- [ ] Add pytorch-lightning training example for GPT
- [ ] Add conversion functions so vision models can be run in the Passio MindsEye app
