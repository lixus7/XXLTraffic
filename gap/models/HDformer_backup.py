import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import random
import math
from torch.distributions.normal import Normal

epsilon = 0.01

LOG2PI = math.log(2 * math.pi)



class SeasonalNorm(nn.Module):
    def __init__(self, cycle_len, cycle_num, d_model, series_num):
        super(SeasonalNorm, self).__init__()
        self.weight = Parameter(torch.randn(cycle_num, cycle_num))
        self.cycle_len = cycle_len
    
    def forward(self, x):
        b, c, n, t = x.shape
        """
        mask = np.zeros((7, 7))
        for i in range(7):
            for j in range(7):
                if np.abs(i - j) < 2:
                    mask[i, j] = 1
        mask = torch.tensor(mask).to(x.device, dtype=torch.float)
        """
        weight = torch.softmax(self.weight, dim=-1)
        #weight = self.weight / ((self.weight ** 2).sum(-1, keepdim=True) + epsilon)** 0.5
        x_cycle = torch.split(x, split_size_or_sections=self.cycle_len, dim=-1)
        x_cycle = torch.stack(x_cycle, -1).permute(0, 1, 2, 4, 3)
        mean_cycle = torch.matmul(weight, x_cycle)
        
        var_cycle = torch.matmul(weight, x_cycle ** 2) - mean_cycle ** 2 + epsilon
        norm_x_cycle = (x_cycle - mean_cycle) / (var_cycle ** 0.5)
        norm_x = norm_x_cycle.reshape(b, c, n, t)
        mean = mean_cycle.reshape(b, -1, n, t)
        var = var_cycle.reshape(b, -1, n, t)
        
        return norm_x, mean, var



class AdaSpatialNorm(nn.Module):
    def __init__(self, series_num):
        super(AdaSpatialNorm, self).__init__()
        #self.adj_mat = Parameter(torch.zeros(series_num, series_num))
        
    def forward(self, x):
        b, c, n, t = x.shape
        x = x + 0.00001* torch.randn(x.shape).to(x.device)
        
        x_norm = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True)
        adj_mat = torch.matmul(x_norm, x_norm.permute(0, 1, 3, 2))
        adj_mat = (adj_mat / t).mean(1, keepdim=True)

        adj_mat = torch.softmax(adj_mat * 5, dim=-1)              
        mean = torch.matmul(adj_mat, x)
        var = torch.matmul(adj_mat, x ** 2) - mean ** 2 + epsilon
        out = (x - mean) / var ** 0.5
        return out, mean, var


class MacroviewNorm(nn.Module):
    def __init__(self, sample_len, d_model):
        super(MacroviewNorm, self).__init__()
        self.sample_len = sample_len
        self.mean_weight =  nn.Conv2d(
            in_channels = d_model,
            out_channels = d_model,
            kernel_size = (1, sample_len))
        self.var_weight =  nn.Conv2d(
            in_channels = d_model,
            out_channels = d_model,
            kernel_size = (1, sample_len))
        
    def forward(self, x):
        b, c, n, t = x.shape
        sample_x = x[..., ::8][..., -self.sample_len:]
        mean = self.mean_weight(sample_x)
        log_prob = - (x - mean) ** 2
        mean = mean.repeat(1, 1, 1, t)
        norm_x = x - mean
        return x, mean, 0, log_prob


class PeriodNorm(nn.Module):
    def __init__(self, period_len):
        super(PeriodNorm, self).__init__()
        self.weight =  Parameter(torch.zeros(1, period_len))
        self.period_len = period_len
        
    def forward(self, x):
        b, c, n, t = x.shape

        x_patch = [x[..., self.period_len-1-i:-i+t] for i in range(self.period_len-1, -1, -1)]
        x_patch = torch.stack(x_patch, dim=-1)
        weights = torch.softmax(self.weight, dim=-1)
        weights = weights.view(1, 1, 1, 1, -1)
        #weight = self.weight / ((self.weight ** 2).sum(-1, keepdim=True) + epsilon)** 0.5
        mean_patch = (weights * x_patch).sum(dim=-1, keepdim=True)
        mean_patch = F.pad(mean_patch.reshape(b * c * n, -1, 1), mode='replicate', pad=(self.period_len-1, 0)).reshape(b, c, n, -1, self.period_len)
        
        var_patch = (weights * x_patch ** 2).sum(dim=-1, keepdim=True)

        var_patch = F.pad(var_patch.reshape(b * c * n, -1, 1), mode='replicate', pad=(self.period_len-1, 0)).reshape(b, c, n, -1, self.period_len)
        var_patch = var_patch - mean_patch ** 2 + epsilon
        
        norm_x_patch = (x_patch - mean_patch) / (var_patch) ** 0.5

        if norm_x_patch.shape[3] > 1:
            norm_x = torch.cat([norm_x_patch[:, :, :, 0], norm_x_patch[:, :, :, 1:, -1]], dim=-1)
            mean = torch.cat([mean_patch[:, :, :, 0], mean_patch[:, :, :, 1:, -1]], dim=-1)
            var = torch.cat([var_patch[:, :, :, 0], var_patch[:, :, :, 1:, -1]], dim=-1)
        else:
            norm_x = norm_x_patch[:, :, :, 0]
            mean = mean_patch[:, :, :, 0]
            var = var_patch[:, :, :, 0]

        
        return norm_x, mean, var


class PeriodExtrapolate(nn.Module):
    def __init__(self, pred_len, input_len):
        super(PeriodExtrapolate, self).__init__()
        self.pred_len = pred_len
        self.input_len = input_len
        self.weight = Parameter(torch.zeros(pred_len, input_len))
        
    def forward(self, x):
        x_input = x[..., -self.input_len:]
        weight = torch.softmax(self.weight, dim=-1)
        proj = torch.matmul(weight, x_input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_proj = torch.cat([x, proj], dim=-1)
        return x_proj


class SeasonalExtrapolate(nn.Module):
    def __init__(self, pred_len, cycle_len, cycle_num):
        super(SeasonalExtrapolate, self).__init__()
        self.pred_len = pred_len
        self.cycle_num = cycle_num
        self.cycle_len = cycle_len
        self.weight = Parameter(torch.zeros(pred_len // cycle_len + 1, cycle_num))
        
    def forward(self, x):
        b, c, n, t = x.shape
        #weight = torch.zeros(self.pred_len // self.cycle_len + 1, self.cycle_num).cuda()
        #weight[:, -1] = 1000
        weight = torch.softmax(self.weight, dim=-1)  
        x_cycle = torch.split(x, split_size_or_sections=self.cycle_len, dim=-1)
        x_cycle = torch.stack(x_cycle, -1)
        proj_cycle = torch.matmul(weight, x_cycle.permute(0, 2, 3, 4, 1))
        x_proj = torch.cat([x_cycle.permute(0, 2, 3, 4, 1), proj_cycle], dim=-2).permute(0, 4, 1, 3, 2).reshape(b, c, n, -1)[...,: t + self.pred_len]
        #x_proj = torch.matmul(weight, x_input.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        
        return x_proj


class PolynomialRegression(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(PolynomialRegression, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  bias=True)
        self.conv_2 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  bias=True)
        self.conv_3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=(1, kernel_size),
                                  bias=True)
        self.conv_4 = nn.Conv2d(in_channels=out_channels,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  bias=True)
        
    
    def forward(self, x):
        x_1 = self.conv_1(x)
        x_2 = self.conv_2(x)
        x_3 = self.conv_3(x)
        
        x_z = self.conv_4(x_1 * x_2) + x_3
        
        return x_z
        

def ConstantExtrapolate(x, num_pred):
    b, c, n, t = x.shape
    x_proj = F.pad(x.reshape(b * c, n, -1), mode='replicate', pad=(0, num_pred)).reshape(b, c, n, -1)
    return x_proj


def ZeroExtrapolate(x, num_pred):
    b, c, n, t = x.shape
    x_proj = F.pad(x.reshape(b * c, n, -1), mode='constant', value=0, pad=(0, num_pred)).reshape(b, c, n, -1)
    return x_proj

class ResidualExtrapolate(nn.Module):
    def __init__(self, d_model, input_len, output_len):
        super(ResidualExtrapolate, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.regreesor = nn.Conv2d(in_channels=d_model, out_channels=d_model * output_len, kernel_size=(1, input_len))
                
    def forward(self, x):
        b, c, n, t = x.shape
        proj = self.regreesor(x[..., ::12][..., -self.input_len:]).reshape(b, -1, c, n).permute(0, 2, 3, 1)
        x_proj = torch.cat([x, proj], dim=-1)
        return x_proj


class MacroExtrapolate(nn.Module):
    def __init__(self, d_model, output_len):
        super(MacroExtrapolate, self).__init__()
        self.regreesor = nn.Conv2d(in_channels=d_model, out_channels=d_model * output_len, kernel_size=1)
        
    def forward(self, x):
        b, c, n, t = x.shape
        proj = self.regreesor(x[..., -1:]).reshape(b, -1, c, n).permute(0, 2, 3, 1)
        x_proj = torch.cat([x, proj], dim=-1)
        return x_proj

class ResidualExtrapolate1(nn.Module):
    def __init__(self, d_model, input_len, output_len):
        super(ResidualExtrapolate1, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.regreesor = nn.Conv2d(in_channels=d_model, out_channels=d_model * output_len, kernel_size=(1, input_len))
        
    def forward(self, x):
        b, c, n, t = x.shape
        proj = self.regreesor(x[..., -self.input_len:]).reshape(b, -1, c, n).permute(0, 2, 3, 1)
        x_proj = torch.cat([x, proj], dim=-1)
        return x_proj
"""
class EncoderBlock(nn.Module):
    def __init__(self, d_in, d_out, seq_len, pred_len, cycle_len, short_period_len, series_num, kernel_size, long_term=1, short_term=1, seasonal=1, spatial=1, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.kernel_size = kernel_size
        self.pred_len = pred_len
        self.series_num = series_num
        self.cycle_len = cycle_len
        self.short_period_len = short_period_len
        self.seq_len = seq_len
        self.long_term = long_term
        self.seasonal = seasonal
        self.short_term = short_term
        self.spatial = spatial
        self.layers = nn.ModuleList()
        for i in range(2):
            self.layers.append(EncoderLayer(d_in, d_out, seq_len, pred_len, cycle_len, short_period_len, series_num, long_term, short_term, seasonal, spatial, dropout))
        
        self.skip_conv = nn.Conv2d(in_channels=d_out,
                                  out_channels=d_out,
                                  kernel_size=1,
                                  bias=True)
        self.residual_conv = nn.Conv2d(in_channels=d_out,
                                  out_channels=d_in,
                                  kernel_size=1,
                                  bias=True)
        self.poly = PolynomialRegression(4 * d_in, d_out, kernel_size)
    

    def forward(self, x):
        b, c, n, t = x.shape
        residuals = []
        structure_components = []
        for i in range(4):
            residual, structure_component = self.layers[i](x)
            x = residual[..., :self.seq_len]
            residuals.append(residual)
            structure_components.append(structure_component)
        
        
        x = torch.cat(structure_components+residuals, dim=1)
        x = F.pad(x,  mode='constant', pad=(self.kernel_size-1, 0))
        x = self.poly(x)
        x_z = x[...,:-self.pred_len]
        s = x[...,-self.pred_len:]
        x_z = self.residual_conv(x_z)                
        s = self.skip_conv(s)
        return x_z, s

class EncoderLayer(nn.Module):
    def __init__(self, d_in, d_out, seq_len, pred_len, cycle_len, short_period_len, series_num, long_term=1, short_term=1, seasonal=1, spatial=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.pred_len = pred_len
        self.series_num = series_num
        self.cycle_len = cycle_len
        self.short_period_len = short_period_len
        self.seq_len = seq_len
        self.long_term = long_term
        self.seasonal = seasonal
        self.short_term = short_term
        self.spatial = spatial
        if long_term: 
            self.long_period_norm = PeriodNorm(seq_len)
        if seasonal:
            self.cycle_norm = SeasonalNorm(cycle_len, seq_len // cycle_len, d_in, series_num)
            self.cycle_ext = SeasonalExtrapolate(pred_len, cycle_len, seq_len // cycle_len)
            self.cycle_residual_ext = SeasonalExtrapolate(pred_len, cycle_len, seq_len // cycle_len)
            #self.cycle_residual_ext = ResidualExtrapolate1(d_in, short_period_len, pred_len)
        
        if short_term:
            self.short_period_norm = PeriodNorm(short_period_len)
            self.short_period_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            self.short_period_residual_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            #self.short_period_residual_ext = ResidualExtrapolate1(d_in, short_period_len, pred_len)
        
        if spatial:
            self.spatial_norm = AdaSpatialNorm(series_num)
            self.spatial_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            self.spatial_residual_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            #self.spatial_residual_ext = ResidualExtrapolate1(d_in, short_period_len, pred_len)
        num_components = long_term + seasonal + short_term + spatial
        self.weights = Parameter(torch.zeros(num_components))
    

    def forward(self, x):
        b, c, n, t = x.shape
        residuals = []
        structure_components = []
        
        ys = []
        x_aux = []
        if self.long_term:
            long_residual, long_mean, _ = self.long_period_norm(x)
            
            long_residual_proj = ConstantExtrapolate(long_residual, self.pred_len)      
            long_mean_proj = ConstantExtrapolate(long_mean, self.pred_len)
            residuals.extend([long_residual_proj])
            structure_components.extend([long_mean_proj])
        
        
        if self.seasonal:
            cycle_residual, cycle_mean, cycle_var = self.cycle_norm(x)
            cycle_mean_proj = self.cycle_ext(cycle_mean)
            cycle_var_proj = self.cycle_ext(cycle_var)
            cycle_residual_proj = self.cycle_residual_ext(cycle_residual)
            #cycle_mean_proj = torch.cat([cycle_mean, cycle_mean_proj], dim=-1)
            structure_components.extend([cycle_mean_proj])
            residuals.extend([cycle_residual_proj])
        
        
        if self.short_term:
            short_residual, short_mean, _ = self.short_period_norm(x)
            short_mean_proj = self.short_period_ext(short_mean)
            short_residual_proj = self.short_period_residual_ext(short_residual)
            if self.pred_len > self.short_period_len:
                short_mean_proj = ZeroExtrapolate(short_mean_proj, self.pred_len - self.short_period_len)
                short_residual_proj = ZeroExtrapolate(short_residual_proj, self.pred_len - self.short_period_len)
            residuals.extend([short_residual_proj])
            structure_components.extend([short_mean_proj])
        
        if self.spatial:
            spatial_reisudal, spatial_mean, spatial_var = self.spatial_norm(x)
            spatial_reisudal_proj = self.spatial_residual_ext(spatial_reisudal)
            spatial_mean_proj = self.spatial_ext(spatial_mean)
            if self.pred_len > self.short_period_len:
                spatial_reisudal_proj = ZeroExtrapolate(spatial_reisudal_proj, self.pred_len - self.short_period_len)
                spatial_mean_proj = ZeroExtrapolate(spatial_mean_proj, self.pred_len - self.short_period_len)
            residuals.extend([spatial_reisudal_proj])
            structure_components.extend([spatial_mean_proj])
        weight = torch.softmax(self.weights, dim=0)
        residual = (torch.stack(residuals, dim=-1) * weight.reshape(1, 1, 1, 1, -1)).sum(-1)
        structure_component = (torch.stack(structure_components, dim=-1) * weight.reshape(1, 1, 1, 1, -1)).sum(-1)
        
        return residual, structure_component
"""
class EncoderLayer(nn.Module):
    def __init__(self, d_in, d_out, seq_len, pred_len, cycle_len, short_period_len, series_num, kernel_size, long_term=1, short_term=1, seasonal=1, spatial=1, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.kernel_size = kernel_size
        self.pred_len = pred_len
        self.series_num = series_num
        self.cycle_len = cycle_len
        self.short_period_len = short_period_len
        self.seq_len = seq_len
        self.long_term = long_term
        self.seasonal = seasonal
        self.short_term = short_term
        self.spatial = spatial
        if long_term: 
            self.long_period_norm = PeriodNorm(seq_len)
            #self.long_period_residual_ext = ResidualExtrapolate(d_in, short_period_len, pred_len)
        if seasonal:
            self.cycle_norm = SeasonalNorm(cycle_len, seq_len // cycle_len, d_in, series_num)
            self.cycle_ext = SeasonalExtrapolate(pred_len, cycle_len, seq_len // cycle_len)
            self.cycle_residual_ext = SeasonalExtrapolate(pred_len, cycle_len, seq_len // cycle_len)
            #self.cycle_residual_ext = ResidualExtrapolate1(d_in, short_period_len, pred_len)
        
        if short_term:
            self.short_period_norm = PeriodNorm(short_period_len)
            self.short_period_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            self.short_period_residual_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            #self.short_period_residual_ext = ResidualExtrapolate1(d_in, short_period_len, pred_len)
        
        if spatial:
            self.spatial_norm = AdaSpatialNorm(series_num)
            self.spatial_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            self.spatial_residual_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
            #self.spatial_residual_ext = ResidualExtrapolate1(d_in, short_period_len, pred_len)
        self.macro_norm = MacroviewNorm(short_period_len, d_in)
        self.macro_ext = MacroExtrapolate(d_in, pred_len)
        self.macro_residual_ext = PeriodExtrapolate(min(pred_len, short_period_len), short_period_len)
        
        self.skip_conv = nn.Conv2d(in_channels=d_out,
                                  out_channels=d_out,
                                  kernel_size=1,
                                  bias=True)
        self.residual_conv = nn.Conv2d(in_channels=d_out,
                                  out_channels=d_in,
                                  kernel_size=1,
                                  bias=True)
        num_components = long_term + seasonal + short_term + spatial
        self.poly = PolynomialRegression((num_components * 2) * d_in, d_out, kernel_size)
    

    def forward(self, x):
        b, c, n, t = x.shape
        residual = x
        xs = []
        structure_xs = []
        
        ys = []
        x_aux = []
        if self.long_term:
            x, long_period_mean, _ = self.long_period_norm(x)
            #x_proj = self.long_period_residual_ext(x)
            
            x_proj = ConstantExtrapolate(x, self.pred_len)      
            long_period_mean_proj = ConstantExtrapolate(long_period_mean, self.pred_len)
            xs.extend([x_proj])
            structure_xs.extend([long_period_mean_proj])
        
        
        if self.seasonal:
            x, cycle_mean, cycle_var = self.cycle_norm(x)
            cycle_mean_proj = self.cycle_ext(cycle_mean)
            cycle_var_proj = self.cycle_ext(cycle_var)
            x_proj = self.cycle_residual_ext(x)
            #cycle_mean_proj = torch.cat([cycle_mean, cycle_mean_proj], dim=-1)
            structure_xs.extend([cycle_mean_proj])
            xs.extend([x_proj])
                
        if self.short_term:
            x, short_period_mean, _ = self.short_period_norm(x)
            short_period_mean_proj = self.short_period_ext(short_period_mean)
            x_proj = self.short_period_residual_ext(x)
            if self.pred_len > self.short_period_len:
                short_period_mean_proj = ZeroExtrapolate(short_period_mean_proj, self.pred_len - self.short_period_len)
                x_proj = ZeroExtrapolate(x_proj, self.pred_len - self.short_period_len)
            xs.extend([x_proj])
            structure_xs.extend([short_period_mean_proj]) 
        reg_loss = 0
        """
        x, macro_mean, _, log_prob = self.macro_norm(x)
        reg_loss = log_prob.mean()
        macro_mean_proj = self.macro_ext(macro_mean)
        x_proj = self.macro_residual_ext(x)
        if self.pred_len > self.short_period_len:
            x_proj = ZeroExtrapolate(x_proj, self.pred_len - self.short_period_len)
        xs.extend([x_proj])
        structure_xs.extend([macro_mean_proj]) 
        """
        if self.spatial:
            x, spatial_mean, spatial_var = self.spatial_norm(x)
            x_proj = self.spatial_residual_ext(x)
            spatial_mean_proj = self.spatial_ext(spatial_mean)
            if self.pred_len > self.short_period_len:
                x_proj = ZeroExtrapolate(x_proj, self.pred_len - self.short_period_len)
                spatial_mean_proj = ZeroExtrapolate(spatial_mean_proj, self.pred_len - self.short_period_len)
            xs.extend([x_proj])
            structure_xs.extend([spatial_mean_proj])

        x = torch.cat(structure_xs+xs, dim=1)
        x = F.pad(x,  mode='constant', pad=(self.kernel_size-1, 0))
        x = self.poly(x)
        x_z = x[...,:-self.pred_len]
        s = x[...,-self.pred_len:]
        x_z = self.residual_conv(x_z)                
        s = self.skip_conv(s)
        
        return x_z, s, reg_loss


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.cycle_len = configs.cycle_len
        self.lookback_len = configs.lookback_len
        self.enc_layers = nn.ModuleList()
        self.ext = configs.ext
        self.start_conv = nn.Conv2d(in_channels=1,
                                    out_channels=configs.d_model,
                                    kernel_size=1)
        self.e_layers = configs.e_layers
        self.global_weight = Parameter(torch.randn(configs.seq_len // configs.cycle_len, 1))
        
        for i in range(configs.e_layers):
            self.enc_layers.append(EncoderLayer(configs.d_model, configs.d_model, configs.seq_len, configs.pred_len, configs.cycle_len, configs.short_period_len, configs.enc_in, configs.kernel_size, configs.long_term, configs.short_term, configs.seasonal, configs.spatial, dropout=0.1))
        self.end_conv = nn.Conv1d(in_channels=configs.d_model * configs.pred_len,
                              groups=configs.pred_len,
                              out_channels=configs.pred_len,
                              kernel_size=1,
                              bias=True)
            
        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        input = x_enc.permute(0, 2, 1).unsqueeze(1)
        in_len = input.size(3)
        b, c, n, t = input.shape
        x = self.start_conv(input)
        #x = input
                
        out = 0
        s = 0
        reg_loss = 0
        for i in range(self.e_layers):
            residual = x
            x, s, log_prob = self.enc_layers[i](x)
            x = x + residual 
            out = out + s
            reg_loss += log_prob
            """
            if i == 1:
                b, c, n, L = x.shape
                #global_pattern = torch.matmul(x.reshape(b, c, n, -1, self.cycle_len).permute(0, 1, 2, 4, 3), torch.softmax(self.global_weight, dim=0)).squeeze(-1)
                x = x[..., -96:]
            if i == 2:
                x = x[..., -48:]
            """
        
        dec_out = self.end_conv(out.permute(0, 3, 1, 2).reshape(b, -1, n))
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        if self.training:
            return dec_out, reg_loss
        else:
            return dec_out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if isinstance(param, Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                pass
      
