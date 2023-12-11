from .arch_part import *
from .norm_part import LayerNorm2d
from .gumbel import *
import numpy as np
import cv2
from torch.distributions import Categorical

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class FourierCNN(nn.Module):
    def __init__(self):
        super(FourierCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1) 
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        promt = self.relu(self.conv3(x)) 
        return promt

# Fourier transform preprocessing module
def batch_fft_preprocess(img_batch_numpy):

    # Convert to numpy array for FFT, but first make sure to process each image
    batch_magnitude = []
    for img_numpy in img_batch_numpy:

        img = np.transpose(img_numpy, (1, 2, 0))  # Move the channel dimension to the end
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        img_np = np.uint8(gray_img * 255)  # Make sure the image is an 8-bit unsigned intege

        f = np.fft.fft2(img_np)  # Note: For a single channel image
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)
        batch_magnitude.append(magnitude_spectrum)
        
    magnitude_array = np.stack(batch_magnitude, axis=0)
    magnitude_tensor = torch.from_numpy(magnitude_array).float().cuda().unsqueeze(1)  # Add channel dimensions
    return magnitude_tensor
 

class gnconv(nn.Module):
    def __init__(self, dim, out_dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)
        self.proj_in_mf = nn.Conv2d(2*dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, out_dim, 1)
        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s

    def forward(self, inform, mask=None, dummy=False):
        x, dpromt = inform[0], inform[1]
        B, C, H, W = x.shape
        fused_x = self.proj_in_mf(torch.cat((x.contiguous(),dpromt.contiguous()),dim=1))
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            if i%3==0:
                x = self.pws[i](x) + dw_list[i+1] #element_wise_product
            elif i%3==1:
                x = self.pws[i](x) + dw_list[i+1] 
            else:
                x = self.pws[i](x) + dw_list[i+1] 

        x = self.proj_out(x)

        return (x, dpromt)

class crCrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        self.d_model=d_model
        self.nhead=nhead
        super(crCrossAttention, self).__init__()

        # Nonlinear projection
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1)

        self.attn = nn.MultiheadAttention(d_model, nhead)  
        
    def forward(self, q, k, v):
        v = self.conv1(v)
        v = self.conv2(v)
        v = self.conv3(v)

        height = v.shape[2]
        width = v.shape[3]

        q = q.view(q.size(0), -1, self.d_model).transpose(0, 1) 
        k = k.view(k.size(0), -1, self.d_model).transpose(0, 1)
        v = v.view(v.size(0), -1, self.d_model).transpose(0, 1)

        out, _ = self.attn(q, k, v)
        out = out.transpose(0, 1).reshape(-1, 512, height, width)
        return out

class stCrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        self.d_model=d_model
        self.nhead=nhead
        super(stCrossAttention, self).__init__()
        
        # Nonlinear projection
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.conv3 = nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  

        # The convolution layers is used to adjust the shape of feature
        self.reshapeOperation = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1), 
        )

        self.attn = nn.MultiheadAttention(d_model, nhead)  
        
    def forward(self, q, k, v):

        q = self.conv1(q)
        q = self.conv2(q)
        q = self.conv3(q)

        height = q.shape[2]
        width = q.shape[3]

        q = q.view(q.size(0), -1, self.d_model).transpose(0, 1)  # shape: (seq_len, batch, features)
        k = k.view(k.size(0), -1, self.d_model).transpose(0, 1)
        v = v.view(v.size(0), -1, self.d_model).transpose(0, 1)

        out, _ = self.attn(q, k, v)
  
        out = out.transpose(0, 1).reshape(-1, 3, height, width)
        out = self.reshapeOperation(out)
        return out



class D3Net(nn.Module):
    def __init__(self, n_channels=3, out_channels=3, num_adb_blocks=12, bilinear=False):
        super(D3Net, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, out_channels))

        self.DA = FourierCNN()
        self.DDB = DDB(inChannels=128, filters=128, num_res_blocks=num_adb_blocks)
        self.crAttention = crCrossAttention(d_model=512, nhead=1)
        self.stAttention = stCrossAttention(d_model=512, nhead=1)


    def forward(self, x):

        #CDDA
        #-------------#
        x_np =  x.cpu().numpy()  
        x_np = batch_fft_preprocess(x_np)  # Extract frequency domain features
        Df = self.DA(x_np)
        crPromt = self.crAttention(Df, Df, x)
        stPromt = self.stAttention(Df, x, x)
        #-------------#
        x1 = self.inc(x)
        x2 = self.down1(x1)

        
        x2 = self.DDB(x2, crPromt, stPromt) # Degradation Decomposition Branch
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
       
        #-------------#

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class adaptiveDecompositionBlock(nn.Module):

    def __init__(self, filters, res_scale=0.2):
        super(adaptiveDecompositionBlock, self).__init__()
        self.res_scale = res_scale
        self.filters = filters
        def block(in_features, non_linearity=True):
            layers = [gnconv(in_features, filters, order=5, gflayer=None, h=14, w=8, s=1.0)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=filters)
        self.b2 = block(in_features=filters)

        self.ln1 = LayerNorm2d(filters) 
        self.ln2 = LayerNorm2d(filters) 
        
        self.conv_1_1 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv_1_2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.beta = nn.Parameter(torch.zeros((1, filters, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, filters, 1, 1)), requires_grad=True)

    def forward(self, x, dpromt):

        inputs, dpromt = x, dpromt

        inputs = self.ln1(inputs)

        x, dpromt = self.b1((inputs, dpromt))
        x = nn.LeakyReLU()(x)
        y = inputs + x * self.beta
        
        x = self.ln2(x)
        x = self.conv_1_1(x)
        x, dpromt = self.b2((x, dpromt))
        x = nn.LeakyReLU()(x)
        y = y + x * self.gamma
        return y  



class decisionUnit(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(decisionUnit, self).__init__()
        self.res_scale = nn.Parameter(torch.Tensor([res_scale]))
        self.ADB = adaptiveDecompositionBlock(filters)

        self.decisionNetwork = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)), 
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(8, 2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, inform):

        x = inform[0]
        
        cPromt = inform[1]
        rPromt = inform[2]

        pInform = torch.add(rPromt, x)
        logits = self.decisionNetwork(pInform)

        temperature = 1  # Adjust the temperature value
        fc_result = F.softmax(logits/temperature, dim=1) # Uniform distribution [0-1]
        fc_result = fc_result.view(-1,2)

        # Apply Gumbel Softmax
        fc_gumbel_result = gumbel_softmax(fc_result, temperature=1)

        # Use the Heaviside step function to determine if the dense_blocks should be executed.
        decisions = heaviside(fc_gumbel_result[:, 1], tau=0.5)  # Threshold tau is 0.5

        # Separating indices
        active_indices = torch.nonzero(decisions > 0.5, as_tuple=True)[0]
        inactive_indices = torch.nonzero(decisions <= 0.5, as_tuple=True)[0]

        if len(active_indices) > 0:
            active_samples = x[active_indices]
            active_cPromt = cPromt[active_indices]
            processed_active_samples = self.ADB(active_samples, active_cPromt)
            processed_active_samples = processed_active_samples.mul(self.res_scale) + active_samples
        else:
            processed_active_samples = torch.Tensor().to(x.device)  # Empty tensor for concatenation
        
        # The inactive samples are unchanged
        inactive_samples = x[inactive_indices]

        # Concatenate the processed and unprocessed samples
        # Note that torch.cat is used here instead of a loop to concatenate results
        all_samples = torch.cat([processed_active_samples, inactive_samples], dim=0)
        
        # We now need to reorder all samples to match the input order
        # First, prepare an index map
        all_indices = torch.cat([active_indices, inactive_indices])
        _, inverse_indices = torch.sort(all_indices)

        # Reorder the samples to match the original order
        final_output = all_samples[inverse_indices]

        return (final_output, cPromt, rPromt)


class DDB(nn.Module):
    def __init__(self, inChannels=128, filters=128, num_res_blocks=12):
        super(DDB, self).__init__()
        self.num_res_blocks = num_res_blocks
        # First layer
        self.conv1 = nn.Conv2d(inChannels, filters, kernel_size=3, stride=1, padding=1)

        # Residual blocks
        self.decisionUnits = nn.Sequential(*[decisionUnit(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks

        # Conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=filters),
            nn.LeakyReLU()
        )
        
        # Conv3
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=filters),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=filters),
            nn.LeakyReLU()
        )

        self.final_conv = nn.Conv2d(filters, inChannels, kernel_size=1, stride=1)

        # Create a transposed convolutional laye
        self.tConv = nn.ConvTranspose2d(in_channels=512, 
                                            out_channels=128, 
                                            kernel_size=4, 
                                            stride=2, 
                                            padding=1)
        self.t2Conv = nn.ConvTranspose2d(in_channels=512, 
                                            out_channels=128, 
                                            kernel_size=4, 
                                            stride=2, 
                                            padding=1)


    def forward(self,x, cPromt, rPromt):
  
        '''promt processing branch'''
        cPromt = self.tConv(cPromt)
        rPromt = self.t2Conv(rPromt)

        x = x
        out1 = self.conv1(x)

        out, cPromt, rPromt = self.decisionUnits((out1, cPromt, rPromt)) 
    
        out_1 = torch.add(out, out1)

        out2 = self.conv2(out_1)
        out = torch.add(out_1, out2)
        out = self.conv3(out)

        out_residual = self.final_conv(out) 
        out_final = torch.add(out_residual, out) # Add the original input (assuming out1 is the original input)
        
        
        
        return out_final
