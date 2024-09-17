function [NORClass,NORYout]=NORMAClassifierNoVTrick(NORClass,sample,varargin)
%      NORMA,不使用V-trick
%      NORClass.f            分类器初值，默认0.
%      NORClass.rho          margin的初值，默认0.
%              .eta          步长eta前面的系数
%              .b            分类器偏移量随机产生。
%              .reg          不采用正则化取值为0，若正则化取值为1，默认0.
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
%            NORClass=kernelapsm([],{x(1,:),1});        % 第1个训练样本
%            NORClass=kernelapsm(NORClass,{x(2,:),-1}); % 第2个训练样本
%            ...
%

% Ref:Online learning with kernels,2004,IEEE SP

% by SAM
% 2019/02/20

%分类器初始值设定。（如果输入空，则使用如下默认值）
if nargin==0 || isempty(NORClass)
        NORClass=struct('f',0,...       该程序没有用到f，记录它是便于理解
                'rho',0,...             边界初值
                'Coeffeta',1,...        eta系数，通常eta=Coeffeta/sqrt(n)
                'b',0,...               分类器偏移量（当前迭代值）
                'alpha',[],...          分类器系数，存储为列向量。
                'sigma',1,...           对于初始值，默认分类错误，否则有些参数开始就不更新，
                'reg',1,...             正则化参数，0表示不使用正则化，1表示正则化系数lambda为1；
                'kernelsize',2,...      核长
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
        NORYout=fkernel(xn,NORClass.samples,'rbf',NORClass.kernelsize)*NORClass.alpha + NORClass.b;
        return;
end

xn=sample{1}; yn=sample{2};
NORClass.samples = [NORClass.samples;xn];  % 每个特征是一个行。
NORClass.labels = [NORClass.labels;yn];
ttt = length(NORClass.labels);

if ttt == 1
      
        NORClass.eta = NORClass.Coeffeta/sqrt(ttt);
        alphaNew = NORClass.eta*NORClass.sigma*yn;
        
        NORClass.alpha = [(1-NORClass.eta*NORClass.reg)*NORClass.alpha ; alphaNew];
        
        NORClass.b = NORClass.b + alphaNew;
        
       
        
        
end

if ttt > 1
        
        NORClass.eta = NORClass.Coeffeta/sqrt(ttt);
        Gn = fkernel(xn,NORClass.samples(1:end-1,:),'rbf',NORClass.kernelsize);
        gOut = yn * (Gn*NORClass.alpha + NORClass.b);
        
       
        if gOut <= NORClass.rho
                NORClass.sigma = 1;
        else
                NORClass.sigma = 0;
        end
        
        NORClass.eta = NORClass.Coeffeta/sqrt(ttt);
        alphaNew = NORClass.eta*NORClass.sigma*yn;
        NORClass.alpha = [(1-NORClass.eta*NORClass.reg)*NORClass.alpha ; alphaNew];
        
        NORClass.b = NORClass.b + alphaNew;


end








    
    


