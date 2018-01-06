import torch
import numpy as np
from mvcnn import mvcnn
import time, os
from tqdm import tqdm

from util import name_to_array

state_dict = "predictions/14865558/model_25.torch"

def predict(model, name):
    new_test = name_to_array(name, "aps")

    # Imitate a batch
    new_test = np.expand_dims(new_test, 0)
    new_test = torch.Tensor(new_test)
    new_test = new_test.clone().repeat(1,1,3,1,1)
    input_var = torch.autograd.Variable(new_test.cuda(), requires_grad=False)

    accur = 0

    # Run through network
    if type(model) != list:
        model = [model]
    for m in model:
        output = torch.nn.Sigmoid()(m(input_var))
        output = output.data.cpu().numpy()[0]
        accur += output

    return accur / len(model)

def test_model(model, base_dir=None, epoch=0):
    time_str = str(int(time.time()))[2:]
    if base_dir == None:
        base_dir = "predictions/{}".format(time_str)
        os.mkdir(base_dir)
    outfile = open('{}/predictions_{}_{}.csv'.format(base_dir, time_str, epoch), 'w+')
    print('Id,Probability', file=outfile)
    test_names = [filename[:-4] for filename in os.listdir('aps/')]
    for name in tqdm(test_names):
        for bodypart, prob in enumerate(predict(model, name)):
            print("{}_Zone{},{}".format(name, bodypart + 1, prob), file=outfile)


print("Testing model only")
if type(state_dict) != list:
    state_dict = [state_dict]
models = []
print("Loading models... ", end="")
for name in tqdm(state_dict):
    model = mvcnn(17, pretrained=True).cuda()
    model.load_state_dict(torch.load(name))
    model.eval()
    models.append(model)
    print("Added {}".format(name))
print("All models loaded. \n Testing... ", end="")
test_model(models)
print("Testing Finished.")
