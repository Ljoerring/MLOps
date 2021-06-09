from model import MyAwesomeModel
import torch

model = MyAwesomeModel()
test_data = torch.rand((64,1,28,28))
def test_model_output():
    output = model(test_data)
    # test dimension of output 
    print(output.shape)
    assert output.shape == (64,10)

test_model_output()




