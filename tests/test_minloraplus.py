import torch
from minloraplus import add_lora, apply_to_lora, disable_lora, enable_lora, get_lora_params, merge_lora, name_is_lora, remove_lora, load_multiple_lora, select_lora, get_lora_state_dict
_ = torch.set_grad_enabled(False)

def test_addlora():
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=5, out_features=7),
        torch.nn.Linear(in_features=7, out_features=3),
    )

    x = torch.randn(1, 5)
    Y0 = model(x)

    add_lora(model)
    y = model(x)
    assert torch.allclose(y, Y0)