import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt

from torch2trt import torch2trt


class ExposureNet(nn.Module):
    def __init__(self, image_height, image_width):
        self._image_height = image_height
        self._image_width = image_width
    
        super(ExposureNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 4 * 4, 2)

    def forward(self, x):
        # feed forward through conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        # flatten
        x = x.view(-1, 64 * 4 * 4)
        
        # feed forward through fully connected layer and a softmax layer
        x = self.fc(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def create_trt_network(trt_model, builder):
    pass

with open(r'exposure_factor_relu_best.pkl', 'rb') as torch_model_file:
    torch_model = pickle.load(torch_model_file)

with open(r'test_images_noaugment_100x100.pkl') as test_images_file:
    test_images = pickle.load(test_images_file)

# create example data
x = torch.rand((1, 3, 50, 50)).cuda()

# convert to TensorRT feeding sample data as input
trt_model = torch2trt(torch_model, [x])

print('input names: %s' % trt_model.input_names)
print('output names: %s' % trt_model.output_names)

#with open('exposure_factor_best.engine', 'wb') as f:
#    f.write(trt_model.engine.serialize())
