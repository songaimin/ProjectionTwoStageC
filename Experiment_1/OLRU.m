function [OL,PAYout]=OLRU(OL,sample,varargin)
% 2023/03/28
% PAYout=fkernel(xn',OL.InputX,'rbf',OL.kernelsize)*OL.Gamma;

%分类器初始值设定。（如果输入空，则使用如下默认值）
if nargin==0 || isempty(OL)
    OL=struct( 'rho',1,...             边界初值
        'Gamma',[],...          分类器系数，存储为列向量。
        'kernelsize',4.1,...      核长
        'n_time',0,...
        'c',1.2,...
        'alpha',0.2,...
        'eta',0.7,...
        'TrainingTime',1500,...
        'InputX',[],...        分类器所用样本，每行为一个样本向量。
        'Labels',[]...         分类器所用样本对应标签，列向量。{±1}
        );
    if nargin==0; return; end
end


% 如果没有标签，则不更新分类器，仅输出分类器结果（这段话为什么一定要放在前面？？？？）
if isnumeric(sample) || (length(sample)==1 && iscell(sample))
    if iscell(sample); xn=sample{1};
    else xn=sample;
    end
    PAYout=fkernel(xn,OL.InputX,'rbf',OL.kernelsize)*OL.Gamma;
    return;
end

xn=sample{1}; yn=sample{2};
OL.n_time = OL.n_time + 1;

if OL.n_time == 1
    OL.Gamma = [OL.Gamma; yn];
    OL.InputX = [OL.InputX;xn];
    OL.Labels = [OL.Labels;yn];
end

Pt = OL.c * OL.TrainingTime^(-OL.alpha);
theta = binornd(1,1-Pt);

if OL.n_time > 1
    Loss = max(0,OL.rho - yn*fkernel(xn,OL.InputX,'rbf',OL.kernelsize)*OL.Gamma);
    
    if Loss > 0 && theta == 0   %犯错
        
        update = OL.eta*Loss*yn;
        OL.Gamma = [OL.Gamma; update];
        OL.InputX = [OL.InputX;xn];
        OL.Labels = [OL.Labels;yn];
    end
    
end


