import torch
import torchvision
import numpy as np
import time
import os
import matplotlib
# Add this to save all the plot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Used to show how much image is loaded.
from tqdm import tqdm
# Shuffel data.
import random
from util import name_to_array, TransformDataset
from mvcnn import mvcnn
from sgdr import CosineLR

DEBUG = True
EPOCH = 100
START = 51
FOLDER = "models/lstm/tmp/"
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
    losses = AverageMeter()

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (batch, target) in enumerate(train_loader):

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
        if (i and i % 64 == 0) or DEBUG:
            print('Epoch/Batch: [{0}/{1}]/[{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, EPOCH, i, len(train_loader), batch_time=batch_time, loss=losses))

        del batch, target, output, loss

    scheduler.step()
    loss_tracker_train.append(losses.avg)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (valdat, target) in enumerate(val_loader):

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
        
        del valdat, target, output, loss

    print('\nVALIDATION \t Time(avg): {batch_time.avg:.3f}\t'
            'Loss(avg): {loss.avg:.4f}'.format(batch_time=batch_time, loss=losses))

    loss_tracker_val.append(losses.avg)

print("Initializing model")
model = mvcnn(17, pretrained=True).cuda()

# Create dictionary matching name to vector
with open('stage1_labels.csv') as train_file:
    train_file.readline()
    # Use dict for easier indexing.
    name_to_vector = {}
    for line in train_file:
        print(line)
        name_zone, label = line.strip().split(',')
        name, zone = name_zone.split('_')
        # Take the number out.
        zone = int(zone[4:])
        if name not in name_to_vector:
            name_to_vector[name] = np.zeros(17)
        name_to_vector[name][zone-1] += int(label)
# If debug, only take 32 images for training.
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
# 16(images for one person) x 1(channel)
training_input = np.empty((sample_cnt, 16, 1, 660, 512), dtype=np.float32)
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
print("There are {} training images with size {}".format(sample_cnt - TEST_CNT, valid_input.shape))

print("Creating DataLoaders...")
training_input = torch.Tensor(training_input)
training_output = torch.Tensor(training_output)
valid_input = torch.Tensor(valid_input)
valid_output = torch.Tensor(valid_output)

dataset = TransformDataset(training_input, training_output, names, train=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=8, pin_memory=True, drop_last=True)
valid_dataset = TransformDataset(valid_input, valid_output, names, train=False)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, num_workers=0, pin_memory=True, drop_last=False)

criterion = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
scheduler = CosineLR(optimizer, step_size_min=1e-4, t0=200, tmult=1)
# Try this for benchmark later.
# scheduler = StepLR(optimizer, step_size = 0.5)

torch.backends.cudnn.benchmark = False

if DEBUG:
    time_str = str(int(time.time()))[2::]
    base_dir = "debug/{}".format(time_str)
else:
    base_dir = "models/lstm"
if not os.path.exists(base_dir):
    os.mkdir(base_dir)
    os.mkdir(base_dir+"/tmp")

loss_tracker_train = []
loss_tracker_val = []
ideal_loss = 0.01 # Ideal.
best_loss = 0xffff
this_loss = 0xffff

if START > 0:
    model.load_state_dict(torch.load(FOLDER + "model_{}.torch".format(START)))
    optimizer.load_state_dict(torch.load(FOLDER + "opt_{}.torch".format(START)))
    with open(FOLDER + 'loss_{}.txt'.format(START)) as loss:
        # Danger but I don't care.
        loss_tracker_train = eval(loss.readline())
        loss_tracker_val = eval(loss.readline())
    print("Old model detected and loaded, resume training from epoch {}".format(START))

print("Beginning training...")
for epoch in range(START+1, EPOCH+1):
    train(data_loader, model, criterion, optimizer, scheduler, epoch)
    validate(valid_data_loader, model, criterion)

    if epoch and epoch % 10 == 0:
        print("Saving Model ... ", end = "")
        torch.save(model.state_dict(), "{}/tmp/model_{}.torch".format(base_dir, epoch))
        torch.save(optimizer.state_dict(), "{}/tmp/opt_{}.torch".format(base_dir, epoch))
        with open('{}/tmp/loss_{}.txt'.format(base_dir, epoch), 'w+') as loss:
            print(loss_tracker_train, file = loss)
            print(loss_tracker_val, file=loss)
        print("Model Saved.")
        print("Plotting train/valid loss ... ", end = "")
        # Save a plot of the average loss over time
        plt.clf()
        plt.plot(loss_tracker_train[1:], label="Training loss")
        plt.plot(loss_tracker_val[1:], label="Validation loss")
        plt.legend(loc="upper left")
        plt.savefig("{}/tmp/predictions_{}.png".format(base_dir, epoch))
        print("Plot Finished.")

    this_loss = loss_tracker_val[-1]
    if this_loss < best_loss:
        best_loss = this_loss
    print("This loss: {}".format(this_loss))
    print("Best loss: {}".format(best_loss), end = '\n\n')
    if this_loss < ideal_loss + 0.0025:
        print("Found better model with {} loss (old loss was {})".format(this_loss, best_loss))
        torch.save(model.state_dict(), "{}/best_model_{}_{:.4f}.torch".format(base_dir, epoch, this_loss))

print("Saving Model ... ", end = "")
torch.save(model.state_dict(), "{}/lstm.torch".format(base_dir, epoch))
torch.save(optimizer.state_dict(), "{}/lstm.torch".format(base_dir, epoch))
print("Model Saved.")
print("Plotting train/valid loss ... ", end = "")
# Save a plot of the average loss over time
plt.clf()
plt.plot(loss_tracker_train[1:], label="Training loss")
plt.plot(loss_tracker_val[1:], label="Validation loss")
plt.legend(loc="upper left")
plt.savefig("{}/predictions_{}.png".format(base_dir, epoch))
print("Plot Finished.")