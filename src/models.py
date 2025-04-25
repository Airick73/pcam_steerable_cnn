import torch
from escnn import gspaces
from escnn import nn

class C8SteerableCNN(torch.nn.Module):
    
    def __init__(self, in_channels=3, n_classes=2):
        """
        E(n)-Equivariant Steerable CNN for PCam dataset with odd spatial dimensions
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            n_classes: Number of output classes (2 for binary classification)
        """
        super(C8SteerableCNN, self).__init__()
        
        # The model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.rot2dOnR2(N=8)
        
        # The input image is now an RGB field (3 channels)
        in_type = nn.FieldType(self.r2_act, in_channels * [self.r2_act.trivial_repr])
        
        # Store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # Convolution 1
        # First specify the output type of the convolutional layer
        # We choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 97, margin=1),  # PCam images are 96x96, pad to 97x97
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # Convolution 2
        in_type = self.block1.out_type
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # Convolution 3
        in_type = self.pool1.out_type
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # Convolution 4
        in_type = self.block3.out_type
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # Modified pool2 to ensure odd output dimensions (adding padding=1)
        self.pool2 = nn.SequentialModule(
            # Adding padding=1 to get odd output dimensions
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2, padding=1)
        )
        
        # Convolution 5
        in_type = self.pool2.out_type
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # Modified pooling layer to ensure odd output dimensions
        self.pool2_5 = nn.SequentialModule(
            # Adding padding=1 to get odd output dimensions
            nn.PointwiseAvgPoolAntialiased(self.block5.out_type, sigma=0.66, stride=2, padding=1)
        )
        
        # Convolution 6 - Modified to maintain odd dimensions
        in_type = self.pool2_5.out_type  # Note: Changed from block5.out_type to pool2_5.out_type
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            # Changed padding from 1 to 2 to maintain odd dimensions
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # Using global pooling to reduce to 1x1
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=1.67, stride=None, padding=0)
        
        self.gpool = nn.GroupPooling(out_type)
        
        # Number of output channels
        c = self.gpool.out_type.size # 64 output channels
        
        # Fully Connected layers
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
    
    def forward(self, input: torch.Tensor):
        # Wrap the input tensor in a GeometricTensor
        x = nn.GeometricTensor(input, self.input_type)
        
        # Check if input has the correct spatial dimensions
        if input.shape[2:] != (97, 97):
            # Automatically pad if necessary
            padder = torch.nn.ZeroPad2d((0, 1, 0, 1))  # pad bottom and right by 1
            input = padder(input)
            # print(f"Input after padding: {input.shape}")
            
            # Re-wrap with correct dimensions
            x = nn.GeometricTensor(input, self.input_type)
        
        # Apply each equivariant block
        # print(f"Input tensor shape: {input.shape}")
        # print(f"After GeometricTensor wrapping: {x.shape}")
        
        x = self.block1(x)
        # print(f"After block1: {x.shape}")
        
        x = self.block2(x)
        # print(f"After block2: {x.shape}")
        
        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")
        
        x = self.block3(x)
        # print(f"After block3: {x.shape}")
        
        x = self.block4(x)
        # print(f"After block4: {x.shape}")
        
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")
        
        x = self.block5(x)
        # print(f"After block5: {x.shape}")

        x = self.pool2_5(x)
        # print(f"After pool2_5: {x.shape}")
            
        x = self.block6(x)
        # print(f"After block6: {x.shape}")
        
        # Pool over the spatial dimensions
        x = self.pool3(x)
        # print(f"After pool3: {x.shape}")
        
        # Pool over the group
        x = self.gpool(x)
        # print(f"After group pooling: {x.shape}")

        # Unwrap the output GeometricTensor
        x = x.tensor
        
        # Classify with the final fully connected layers
        x = self.fully_net(x.reshape(x.shape[0], -1))  
        # print(f"Final output: {x.shape}")
        
        return x