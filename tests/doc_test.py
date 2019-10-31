import torchlayers


def test_basic():
    layer = torchlayers.Conv(64)
    print(layer.__qualname__)
    raise ValueError
    # output = layer(inputs)
    # scripted = torch.jit.script(layer)
    # torch.save(layer, temp / "conv_model.pt")

    # new_layer = torch.load(temp / "conv_model.pt")
    # new_output = new_layer(inputs)
    # assert torch.allclose(output, new_output)
