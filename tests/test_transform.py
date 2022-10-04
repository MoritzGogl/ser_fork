import torch
from ser.transforms import flip
def test_transform():
    img=torch.FloatTensor([[[1,2],[3,4]]])
    expectation=torch.FloatTensor([[[4,3],[2,1]]])
    assert torch.equal(flip()(img),expectation)

