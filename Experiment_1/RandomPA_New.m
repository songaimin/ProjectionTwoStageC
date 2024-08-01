function [PerClass,PerYout]=RandomPA_New(PerClass,sample,varargin)
% 20200215,预测正确不进字典
% 在原始Perceptron_1上修改，随机去掉一个，但非最新的。
% 20200212修改，去掉偏差项b; 去掉步长eat，去掉正则项
%      PerClass.f            分类器初值，默认0.
%      PerClass.rho          margin的初值，默认0.
%              .alpha        分类器系数
%              .samples      分类器所用样本
%              .labels       分类器所用样本对应标签。{±1}
%              .kernelfun    核函数句柄，默认是方差为1的高斯核。
%                            函数形式应为：y=kernelfun(x1,x2)。
%   sample     输入样本。如果为{x,y}型二元元胞数组，则x为样本数据(行向量)，
%              y为标签（±1），同时此数据将用于更新分类器；如果为单一
%              数据向量x或{x}（行向量），则表明仅有样本数据，缺少标签，此时
%              仅输出分类器结果，不更新分类器。
%   输出：
%         y   对输入样本的分类器处理结果（核空间内样本点到超平面的距离）。
%         CLASSIFIER   更新后的分类器（如果输入了训练样本）。
%
%   e.g.
%            PerClass=kernelapsm([],{x(1,:),1});        % 第1个训练样本
%            PerClass=kernelapsm(PerClass,{x(2,:),-1}); % 第2个训练样本
%            ...
%

% Ref:Online learning with kernels,2004,IEEE SP

% by SAM
% 2019/02/20

%分类器初始值设定。（如果输入空，则使用如下默认值）
if nargin==0 || isempty(PerClass)
        PerClass=struct('f',0,...       该程序没有用到f，记录它是便于理解
                'rho',0,...             边界初值
                'alpha',[],...          分类器系数，存储为列向量。
                'L',200,...             Budget长度
                'kernelsize',1,...      核长
                'samples',[],...        分类器所用样本，每行为一个样本向量。
                'labels',[]...         分类器所用样本对应标签，列向量。{±1}
                );
        if nargin==0; return; end
end


% 如果没有标签，则不更新分类器，仅输出分类器结果（这段话为什么一定要放在前面？？？？）
if isnumeric(sample) || (length(sample)==1 && iscell(sample))
        if iscell(sample); xn=sample{1};
        else xn=sample;
        end
        PerYout=fkernel(xn,PerClass.samples,'rbf',PerClass.kernelsize)*PerClass.alpha;
        return;
end

xn=sample{1}; yn=sample{2};



ttt = length(PerClass.alpha) + 1;

if ttt == 1
    PerClass.samples = [PerClass.samples;xn];  
    PerClass.labels = [PerClass.labels;yn];
    PerClass.alpha = [PerClass.alpha ; yn];  
end

if ttt > 1
        

        Gn = fkernel(xn,PerClass.samples,'rbf',PerClass.kernelsize);
        gOut = yn * (Gn*PerClass.alpha);
        
        
        if gOut <= PerClass.rho
            PerClass.samples = [PerClass.samples;xn];
            PerClass.labels = [PerClass.labels;yn];            
            PerClass.alpha = [PerClass.alpha ; yn];
            if length(PerClass.alpha) > PerClass.L
                Index_remove = randi(length(PerClass.alpha)-1,1,1);
                PerClass.alpha(Index_remove) = [];
                PerClass.samples(Index_remove,:) = [];
                PerClass.labels(Index_remove) = [];
            end
            
        end
        

end








    
    


