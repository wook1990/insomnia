import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
from torchvision.models import mobilenet_v2
from pose_dataset import PoseDataset
from torch.utils.data import Dataset, DataLoader
import time


epoch_begin = 0
epoch_end = 1000
batch_size = 43
lr = 0.001
cuda = False
device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hidden_size = 128
output_size = 5
num_layers = 2
infer_batch_size = 10


class PoseRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PoseRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        print('PoseRNN', '[forward]')
        output, _ = self.rnn(input, hidden)
        print('PoseRNN', output.shape)
        output = output[:,-1,:]
        # output = output.contiguous().view(-1, hidden_size)
        print('PoseRNN', output.shape)
        output = self.output_fc(output)
        print('PoseRNN', output.shape)
        output = self.softmax(output)
        print('PoseRNN', output.shape)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


# pose = PoseRNN(53, hidden_size, 5)
# pose = pose.to(device)

# poseNet = nn.RNN(54, hidden_size, 2, batch_first=True)
poseNet = PoseRNN(54, hidden_size)
poseNet = poseNet.to(device)
print(poseNet)

train_dataset = PoseDataset(cuda=cuda)
infer_dataset = PoseDataset(is_train=False, cuda=cuda)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
infer_dataloader = DataLoader(infer_dataset, batch_size=infer_batch_size, shuffle=False)

# bce = torch.nn.BCELoss()
bce = torch.nn.MSELoss()
# bce = torch.nn.CrossEntropyLoss()
# bce = torch.nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(poseNet.parameters(), lr=lr, betas=(0.5, 0.999))


def infer():
    poseNet.eval()

    loss_max = 0.
    worst_item = 0.
    loss_count = 0
    for i, sample in enumerate(infer_dataloader):
        samples = sample[0]
        label = sample[1]
        predict = process_rnn(samples, infer_batch_size)
#     print(predict)

    check_labels = label.max(1)[1]
    check_predict = predict.max(1)[1]
    check_eq = torch.eq(check_labels, check_predict).sum().item()

    loss = bce(predict, label)
    print("infer loss: ", loss.item(), "wrong count: %d" % (label.shape[0]-check_eq))

    poseNet.train()

    print("infer is done")


def process_rnn(samples, batch_size=batch_size):
    hidden = torch.zeros(num_layers, batch_size, hidden_size).to(device)

#     print("\nlabel shape")
#     print(label.shape)
#     print("\nsamples shape")
#     print(samples.shape)
#     print("\nsamples[i] shape")
#     print(samples[0].shape)

    output = poseNet(samples, hidden)
#     print("\noutput shape")
#     print(samples[0].shape)
#     print("\nhidden shape")
#     print(samples[0].shape)

    return output



start_time = time.time()
for e in range(epoch_begin, epoch_end):
  if e%10 == 0 and e>0:
    infer()      

#   if e%20 == 0 and e>0:
#     torch.save(mobileV2.state_dict(), "pose.torch_%04d" % e)

  for i, sample in enumerate(train_dataloader):
    samples = sample[0]
    label = sample[1]
#     print(samples.shape)
#     print(label.shape)
#     print(sample[0][0])

    optimizer.zero_grad()
    
    predict = process_rnn(samples)
#     print("\npredict shape")
#     print(predict.shape)
#     predict = predict[:,-1,:]
#     predict = predict.view(-1, output_size)
#     print("\npredict shape")
#     print(predict.shape)
#     print("\npredict value")
#     print(predict)
    
    loss = bce(predict, label)

    loss.backward()
    optimizer.step()
#     infer()
#     exit(0)
    
#     print(label[0])
#     print(predict[0])
    check_labels = label.max(1)[1]
    check_predict = predict.max(1)[1]
    check_eq = torch.eq(check_labels, check_predict).sum().item()

    passed = time.time() - start_time
    log_format = "Epoch: [%04d], [%04d/%04d] time: %.4f, loss: %.5f, wrong count: %d, predict1: %.4f"
    print(log_format % (e, i, len(train_dataloader) , passed, loss.item(), label.shape[0]-check_eq, predict[0][0].item()))


#   infer()
# torch.save(mobileV2.state_dict(), "pose.torch")


# for iter in range(1, n_iters + 1):
#     category, line, category_tensor, line_tensor = randomTrainingExample()
#     output, loss = train(category_tensor, line_tensor)
#     current_loss += loss
# 
#     # Print iter number, loss, name and guess
#     if iter % print_every == 0:
#         guess, guess_i = categoryFromOutput(output)
#         correct = '✓' if guess == category else '✗ (%s)' % category
#         print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))
# 
#     # Add current loss avg to list of losses
#     if iter % plot_every == 0:
#         all_losses.append(current_loss / plot_every)
#         current_loss = 0
