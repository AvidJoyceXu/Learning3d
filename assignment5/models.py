import torch
import torch.nn as nn
import torch.nn.functional as F

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Classification MLP
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # Convert points to (B, 3, N) for conv1d
        x = points.transpose(2, 1).contiguous()
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global max pooling
        x = torch.max(x, 2)[0]
        
        # MLP for classification
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global feature MLP
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(64)
        
        # Point-wise feature MLP
        self.point_conv1 = nn.Conv1d(320, 64, 1)  # 256 + 64 = 320 channels
        self.point_conv2 = nn.Conv1d(64, 64, 1)
        self.point_conv3 = nn.Conv1d(64, num_seg_classes, 1)
        
        self.point_bn1 = nn.BatchNorm1d(64)
        self.point_bn2 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # Convert points to (B, 3, N) for conv1d
        x = points.transpose(2, 1).contiguous()
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global feature vector
        global_feat = torch.max(x, 2)[0]
        
        # Global feature MLP
        global_feat = F.relu(self.bn4(self.fc1(global_feat)))
        global_feat = self.dropout(global_feat)
        global_feat = F.relu(self.bn5(self.fc2(global_feat)))
        global_feat = self.dropout(global_feat)
        
        # Expand global feature to match point features
        global_feat_expanded = global_feat.unsqueeze(2).expand(-1, -1, x.size(2))
        
        # Concatenate global and point features
        concat_feat = torch.cat([x, global_feat_expanded], 1)
        
        # Point-wise feature MLP
        x = F.relu(self.point_bn1(self.point_conv1(concat_feat)))
        x = F.relu(self.point_bn2(self.point_conv2(x)))
        x = self.point_conv3(x)
        
        # Convert back to (B, N, num_seg_classes)
        x = x.transpose(2, 1)
        
        return x



