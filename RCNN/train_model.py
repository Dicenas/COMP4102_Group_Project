import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import cv2

def apply_lighting_correction(image):
    # Apply histogram equalization
    corrected_image = cv2.equalizeHist(image)
    
    return corrected_image

def apply_noise_reduction(image, kernel_size=(5, 5), sigma_x=0):
    # Apply Gaussian blur for noise reduction
    smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
    
    return smoothed_image

def apply_background_segmentation(image):
    # Apply Otsu's thresholding; returns (threshold used by the function which is computed automatically, thresholded image)
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    return segmented_image

class CustomDataset(Dataset):
    def __init__(self, img_dir, annotation_path, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            annotation_path (string): Path to the COCO-style annotation JSON file.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        # Load annotations
        with open(annotation_path) as f:
            self.annotations = json.load(f)
        self.images = self.annotations['images']
        # Create an index to map image ids to annotation ids
        self.image_id_to_annotations = {}
        for annotation in self.annotations['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(annotation)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Annotations for this image
        image_id = img_info['id']
        annotations = self.image_id_to_annotations.get(image_id, [])
        
        # You may want to convert COCO bounding boxes to your preferred format
        boxes = []
        labels = []
        for ann in annotations:
            # COCO format: [x_min, y_min, width, height]
            x_min, y_min, width, height = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x_min, y_min, x_min + width, y_min + height])
            labels.append(ann['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        #image = apply_background_segmentation(apply_noise_reduction(apply_lighting_correction(image)))

        if self.transform:
            image = self.transform(image)
        
        # Note: Depending on your model, you might need to adjust the format of the returned targets (e.g., for use with torchvision models)
        target = {'boxes': boxes, 'labels': labels}

        return image, target
    

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = CustomDataset(img_dir='',
                              annotation_path='train_annotations.txt',
                              transform=transform)


# Simple CNN Backbone
class Net(nn.Module):
    def __init__(self, num_features=512):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        return x
    


# Assuming a simple dataset and dataloader setup
# dataset = YourCustomDataset()
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Custom R-CNN Model
class CustomRCNN(nn.Module):
    def __init__(self, num_classes):
        super(CustomRCNN, self).__init__()
        self.cnn = Net()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(512 * 7 * 7, num_classes)
        self.box_regressor = nn.Linear(512 * 7 * 7, 4)  # Assuming 4 values for bounding box (x1, y1, x2, y2)

    def forward(self, x):
        x = self.cnn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        class_logits = self.classifier(x)
        bbox_deltas = self.box_regressor(x)
        return class_logits, bbox_deltas


net=CustomRCNN(27)
CUDA=torch.cuda.is_available()
if CUDA:
  net=net.cuda()

# Let's first define our device as the first visible cuda device if we have
# CUDA available:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

"""
# Instantiate the model
model = CustomRCNN(num_classes=21)  # Example: VOC dataset has 20 classes + 1 background

# Example training loop setup
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Placeholder for actual training loop
# for images, targets in dataloader:
#     optimizer.zero_grad()
#     class_logits, bbox_deltas = model(images)
#     loss = compute_loss(class_logits, bbox_deltas, targets)  # You'll need to implement loss calculation
#     loss.backward()
#     optimizer.step()

"""