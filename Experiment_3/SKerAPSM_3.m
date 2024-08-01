function [SKer,yout]=SKerAPSM_3(SKer,sample,varargin)

% 固定步长，mu = 1;
% 原来的程序，下面这个语句错了，因为负负得正
%         SKer.rho=max(0,SKer.nu*(SKer.theta-...
%             -SKer.theta0)+SKer.rho0);  
% 正确的是
%         SKer.rho=max(0,SKer.nu*(SKer.theta-...
%             SKer.theta0)+SKer.rho0); 


if nargin==0 || isempty(SKer)
    SKer=struct('rho0',1,...
        'rho',1,...             边界值（初值与当前迭代值）。rho0默认1.
        'theta0',1,...
        'theta',1,...           边界值调节线性模型自变量。theta0默认1.
        'dtheta',1e-3,...       theta的变化步长。默认1e03.
        'nu',0.1,...            边界值调节线性模型斜率.默认0.1.
        'R',0.5,...             可行度（用于边界值调整）。默认0.5.
        'q',4,...               同时处理的样本数。默认16.
        'b',0,...               分类器偏移量（当前迭代值）
        'muAdj',0.1,...           投影外插因子系数（mu=muAdj*Mn）。如果为1，则迭代时计算.
        'mu',0,...              投影外插因子.默认1.当muAdj=0时有效。
        'fnorm',0,...           当前分类器的二范数
        'delta',5,...           投影闭球半径。默认1.
        'L',700,...             稀疏化后保留的最大样本数。默认100.
        'gamma',[],...          分类器系数，列向量。
        'InputX',[],...         分类器所用样本，每行为一个样本向量。
        'labels',[],...         分类器所用样本对应标签，列向量。{±1}
        'kernelfun',@RBFKernel...   核函数句柄，默认是方差为1的高斯核。
        );
    if nargin==0; return; end   % 这是？？？？？
end

% 参数选项
bexRho=false; % 是否进行边界值的迭代，true表明不迭代，而使用classifier中的rho计算。
bnoSparse=false; % 是否进行稀疏化，true表明不进行稀疏化。否则删除最早的样本进行稀疏化。
bnoBall=false; % 是否投影到闭球，true表明不投影。否则投影到闭球。
if nargin>2
    for n=1:length(varargin)
        switch lower(varargin{n})   % 这是？？？？？
            case 'exrho'; bexRho=true;
            case 'nosparse'; bnoSparse=true;
            case 'bnoball'; bnoBall=true;
        end
    end
end


% 如果没有标签，则不更新分类器，仅输出分类器结果
if isnumeric(sample) || (length(sample)==1 && iscell(sample)) % ???????
    if iscell(sample); xn=sample{1}; %为什么用{}？？？？
    else xn=sample;
    end
    yout=EvalOutput(SKer,xn);
    return;  %%为什return???
end


xn=sample{1}; yn=sample{2};

%计算系数beta.
% 这里输入是sample, 参数是samples, 二者不一样。
n_samples=size(SKer.InputX,1)+1; % 当前样本索引    %为什么不用length，用size，为什么要+1；


if n_samples < SKer.q
    Jn=1:n_samples;
    SKer.w = ones(n_samples,1)/n_samples;
else
    Jn=( n_samples-SKer.q+1 ):n_samples;      % 最近SKer.q个样本索引
    SKer.w=ones(SKer.q,1)/SKer.q;
end


x=[SKer.InputX(Jn(1:end-1),:);xn];  % 最近q个样本
y=[SKer.labels(Jn(1:end-1));yn];
Gn=SKer.kernelfun(x,x); % 最近q个样本的gram矩阵
anorm=1+diag(Gn); % 法向量aj的范数：1+k(xj,xj)(列向量)
if n_samples==1; 
    g=SKer.b; % 分类器的初值为零，因此第一个点在高维空间内积即f(x)也为0.
else
    g=EvalOutput(SKer,x); %最近q个样本的分类器输出
end

beta=max(0,SKer.rho-y.*g).*(SKer.w).*y./anorm; % 用于分类器更新的变量beta（向量）


if SKer.muAdj~=0; % 使用动态调整的mu值
    if SKer.muAdj<0 || SKer.muAdj>2
        error('the parameter ''SKer.muAdj'' must be within the interval, [0,2].');
    end
    An = anorm'*(beta.^2./SKer.w);  %一行乘一列，An是数
    if An==0; % 当前分类器落在各个凸集的交集内
        SKer.mu=0; %把1变成0；其实当An==0时，Mn=0，所以SKer.mu=0，不如直接令SKer.mu=0
    else
%         Bn=beta'*(1+Gn)*beta;
%         Mn=An/Bn;
%         SKer.mu=SKer.muAdj*Mn;  
        SKer.mu = 1;
    end
end



% 更新分类器
SKer.gamma(n_samples,1)=0; %想表达的是gamma^(n_samples)_(n_samples)=0;
SKer.gamma(Jn)=SKer.gamma(Jn)+SKer.mu*beta;
fnorm=SKer.fnorm+2*SKer.mu*beta'...
    *(g-SKer.b)+SKer.mu^2*beta'*Gn*beta; % 分类器范数（高维空间的长度）。


%先投影，后稀疏
if ~bnoBall % 投影到闭球
    sfnorm=sqrt(fnorm);
    if sfnorm>SKer.delta
        SKer.gamma=SKer.gamma*SKer.delta/sfnorm;
        fnorm=SKer.delta^2;
    end
end




if ~bnoSparse % 进行稀疏化
    if n_samples>SKer.L %达到容量，移除最早的样本点，并修改分类器范数。
        obsoleteSample=SKer.InputX(1,:);
        obsoleteGamma=SKer.gamma(1);
        fnorm=fnorm-2*obsoleteGamma*SKer.gamma'...
            *SKer.kernelfun([SKer.InputX;xn],obsoleteSample)...
            +obsoleteGamma^2*SKer.kernelfun(obsoleteSample,obsoleteSample);
        SKer.gamma=SKer.gamma(2:end);
        SKer.InputX=SKer.InputX(2:end,:);
        SKer.labels=SKer.labels(2:end);
    end
end



SKer.b=SKer.b+SKer.mu*sum(beta);
SKer.InputX(end+1,:)=xn;
SKer.labels(end+1,1)=yn;
SKer.fnorm=fnorm;



if ~bexRho; % 更新边界值
    Rfeas=sum(beta==0)/length(beta);
    if Rfeas>=SKer.R
        SKer.theta=SKer.theta + SKer.dtheta;
        SKer.rho=max(0,SKer.nu*(SKer.theta-SKer.theta0)+SKer.rho0);
    else
        SKer.theta=SKer.theta - SKer.dtheta;
        SKer.rho=max(0,SKer.nu*(SKer.theta-SKer.theta0)+SKer.rho0);     
    end
end






function g=EvalOutput(SKer,x)
g=SKer.kernelfun(x,SKer.InputX)...
    *SKer.gamma+SKer.b;



function y=RBFKernel(x1,x2)
sigma=1;
y=fkernel(x1,x2,'rbf',sigma);
