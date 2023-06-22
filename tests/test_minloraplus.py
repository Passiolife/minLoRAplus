import torch
from minloraplus import add_lora, remove_lora, merge_lora, apply_to_lora, disable_lora, enable_lora
_ = torch.set_grad_enabled(False)

model = torch.nn.Sequential(
        torch.nn.Linear(in_features=5, out_features=7),
        torch.nn.Linear(in_features=7, out_features=3),
    )

x = torch.randn(1, 5)
Y0 = model(x)

def test_addlora():

    add_lora(model)
    y = model(x)
    assert torch.allclose(y, Y0)

def test_initB():
    y0 = model(x)
    model.apply(apply_to_lora(lambda x: torch.nn.init.ones_(x.lora_B)))
    y = model(x)

    assert not torch.allclose(y, y0)

def test_disable():
    disable_lora(model)
    y = model(x)
    assert torch.allclose(y, Y0)

def test_enable():
    enable_lora(model)
    y = model(x)
    assert not torch.allclose(y, Y0)

def test_remove_lora():
    remove_lora(model)
    assert torch.allclose(model(x), Y0)

def test_add_merge():
    add_lora(model)
    model.apply(apply_to_lora(lambda x: torch.nn.init.ones_(x.lora_B)))
    merge_lora(model)
    y = model(x)
    assert not torch.allclose(y, Y0)
