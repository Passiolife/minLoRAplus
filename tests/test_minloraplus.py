import torch
from minloraplus.model import add_lora
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