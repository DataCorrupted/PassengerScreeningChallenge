import torch
import torchvision
import torchsample.transforms
import numpy as np
import time
import os
import matplotlib
from tqdm import tqdm
import random
# Add this to save all the plot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import get_x_views, TransformDataset
from skimage import io
from mvcnn import mvcnn
from sgdr import CosineLR

TEST_MODEL = False          # Creates predictions file
DEBUG = True                # L`oads small dataset and plots augmented images for debugging
VIEW_COUNT_TOTAL = 16       # Total number of views in our scans. APS files have 16.
VIEW_COUNT_SAMPLE = 16      # Total number of views sampled from the scan. I now use all 16 and this line isn't needed.
epochs = 25
state_dict = None           # Loads a previous state of the model for picking back up training or making predictions.
opt_dict = None             # Loads a previous state of the optimizer for picking back up training if it was cut short.
TEST_CNT = 1024 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    ''' Learned from '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (batch, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch = batch.cuda(async=True)
        target = target.cuda(async=True)

        # Preloaded resnet-50 requires 3 channel.
        batch = batch.clone().repeat(1, 1, 3, 1, 1)

        batch = torch.autograd.Variable(batch)
        target = torch.autograd.Variable(target)

        # compute output
        output = model(batch)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.data[0], batch.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i and i % 10 == 0:
            print('Epoch - Batch: [{0}/{1}] - [{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, epochs, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        del batch, target, output, loss

    scheduler.step()
    loss_tracker_train.append(losses.avg)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (valdat, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        valdat = valdat.cuda(async=True)
        target = target.cuda(async=True)

        valdat = valdat.clone().repeat(1, 1, 3, 1, 1)
        valdat = torch.autograd.Variable(valdat)
        target = torch.autograd.Variable(target)

        # compute output
        output = model(valdat)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.data[0], valdat.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('VALIDATION:')
        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(batch_time=batch_time,
                data_time=data_time, loss=losses))
        
        del valdat, target, output, loss

    loss_tracker_val.append(losses.avg)

def predict(model, name):
    new_test = name_to_array(name, "test")

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
    test_names = set([filename.split('.')[0] for filename in os.listdir('test/')])
    for name in test_names:
        print(name)
        for bodypart, prob in enumerate(predict(model, name)):
            print("{}_Zone{},{}".format(name, bodypart + 1, prob), file=outfile)

def name_to_array(name, path):
    # Given a name and path, return an array of images.
    array = np.array(get_x_views("{}/{}.{}".format(path, name, "aps"), x=VIEW_COUNT_TOTAL))
    array = np.expand_dims(array, 1)
    array = np.pad(array, ((0,0), (0,0), (0, 1), (0, 0)), mode="constant", constant_values=0)
    
    return array


if TEST_MODEL:
    print("Testing model only")
    models = []
    if type(state_dict) != list:
        state_dict = [state_dict]
    for name in state_dict:
        model = mvcnn(17, pretrained=True).cuda()
        model.load_state_dict(torch.load(name))
        model.eval()
        models.append(model)
        print("Added {}".format(name))
    test_model(models)
    exit()

print("Initializing model")
model = mvcnn(17, pretrained=True).cuda()

# Create dictionary matching name to vector
train_file = open('stage1_labels.csv')
train_file.readline()
# Use dict for easier indexing.
name_to_vector = {}
for line in train_file:
    name_zone, label = line.strip().split(',')
    name, zone = name_zone.split('_')
    # Take the number out.
    zone = int(zone[4:])
    if name not in name_to_vector:
        name_to_vector[name] = np.zeros(17)
    name_to_vector[name][zone-1] += int(label)

# If debug, only take 16 images for training.
if DEBUG:
    name_to_vector = {k: name_to_vector[k] for k in sorted(name_to_vector.keys())[:32]}
    TEST_CNT = 30
name_to_vector = list(name_to_vector.items())
random.shuffle(name_to_vector)
print("Loading Images...")

# Convert to raw training input and output
# Images are (660, 512)
sample_cnt = len(name_to_vector)
print("{} samples found.".format(sample_cnt))

names = [None] * sample_cnt
training_input = np.empty((sample_cnt, VIEW_COUNT_TOTAL, 1, 660  + 1, 512  + 0), dtype=np.float32)
training_output = np.empty((sample_cnt, 17))
for i in tqdm(range(sample_cnt)):
    name, is_danger = name_to_vector[i]
    input_tensor = name_to_array(name, "aps")
    training_input[i] = input_tensor
    training_output[i] = is_danger
    names[i] = name
print("Images Loading Finished.")


print("Splitting into train/validation sets...", end='')
training_input, valid_input = training_input[0:TEST_CNT], training_input[TEST_CNT:]
training_output, valid_output = training_output[0:TEST_CNT], training_output[TEST_CNT:]
print("Split Finished.")
print("There are {} training images with size {}".format(TEST_CNT, training_input.shape))
print("There are {} training images with size {}".format(5, valid_input.shape))

print("Creating DataLoaders...")
training_input = torch.Tensor(training_input)
training_output = torch.Tensor(training_output)
valid_input = torch.Tensor(valid_input)
valid_output = torch.Tensor(valid_output)

dataset = TransformDataset(training_input, training_output, names, train=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, sampler=None, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=True)
valid_dataset = TransformDataset(valid_input, valid_output, names, train=False)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, pin_memory=True, drop_last=False)

criterion = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
scheduler = CosineLR(optimizer, step_size_min=1e-4, t0=200, tmult=1)
# Try this for benchmark later.
# scheduler = StepLR(optimizer, step_size = 0.5)
if state_dict:
    model.load_state_dict(torch.load(state_dict))
    optimizer.load_state_dict(torch.load(opt_dict))
    print("Loaded old weights")

torch.backends.cudnn.benchmark = False

print("Beginning training...")
time_str = str(int(time.time()))[2::]
# Path to save models and predictions
base_dir = "predictions/{}".format(time_str)
if not os.path.exists(base_dir) and not DEBUG:
    os.mkdir(base_dir)

loss_tracker_train = []
loss_tracker_val = []
best_loss = 0.01 # Ideal.
this_loss = 0xffff

for epoch in range(epochs+1):
    train(data_loader, model, criterion, optimizer, scheduler, epoch)
    validate(valid_data_loader, model, criterion)

    if epoch and epoch % 25 == 0 and not DEBUG:

        print("Saving Model ...", end = "")
        torch.save(model.state_dict(), "{}/model_{}.torch".format(base_dir, epoch))
        torch.save(optimizer.state_dict(), "{}/opt_{}.torch".format(base_dir, epoch))
        print("Model Saved.")
        print("Plotting train/valid loss ...", end = "")
        # Save a plot of the average loss over time
        plt.clf()
        plt.plot(loss_tracker_train[1:], label="Training loss")
        plt.plot(loss_tracker_val[1:], label="Validation loss")
        plt.legend(loc="upper left")
        plt.savefig("{}/predictions_{}.png".format(base_dir, epoch))
        print("Plot Finished.")

        print("Predicting...")
        test_model(model, base_dir, epoch)

    this_loss = loss_tracker_val[-1]
    print("This loss: {}".format(this_loss))
    print("Best loss: {}".format(best_loss))

    if this_loss < best_loss + 0.0025:
        print("Found better model with {} loss (old loss was {})".format(this_loss, best_loss))
        best_loss = min(this_loss, best_loss)
        torch.save(model.state_dict(), "{}/best_model_{}_{:.4f}.torch".format(base_dir, epoch, this_loss))