function k=fkernel(x,y,varargin)
% K = FKERNEL(X,Y,TYPE)
% evaluates kernel function on X and Y.
%
% If X and Y are values (or row vectors) the can be evaluated on the kernel
% function, then K is a scalar denoting the result.
%
% If X and Y are column vectors whose elements can be evaluated on the
% kernel function, and X is denoted by [x1,x2,...,xn]', Y is denoted by
% [y1,y2,...,ym], the K is the gram matrix of X and Y, where if the kernel
% function is denoted by k, then (K)ij = k(xi,yj), where i and j are the
% row and column entries of K. Note that the elements of X and Y can be row
% vectors, only if they can be evaluated on the kernel function.
% That is, 
%          X=[x1,x2,...,xn]';
%          Y=[y1,y2,...,ym]';
%          K=fkernel(X,Y);
% then,    
%          K=[k(x1,y1),k(x1,y2),...,k(x1,ym);
%             k(x2,y1),k(x2,y2),...,k(x2,ym);
%             ...
%             k(xn,y1),k(xn,y2),...,k(xn,ym)];
% Note that, xi and yi can be column vectors.
%
% K = FKERNEL(X,Y,TYPE,PARA1,PARA2,...). PARA* are the extra parameters
% needed to evaluate the kernel function specified by TYPE.
%
% TYPE is the type of the kernel function. It can be 
%       'rbf'           radial basis function: exp(-||x-y||^2/(2*sigma^2))
%                       e.g. fkernel(x,y,'rbf',sigma);
%       'linear'        linear kernel: x'y
%       'poly'          polynominal kernel: (C+x'y)^n
%                       e.g. fkernel(x,y,'poly',C,n);
%       'exp'           exponential kernel: exp(x'y/2/sigma^2)
%                       e.g. fkernel(x,y,'exp',sigma);
%       'tanh' or 'sigmoid'   sigmoid kernel: tanh(b+a*x'y)
%                             e.g. fkernel(x,y,'tanh',a,b);

% by mz.
% 09/01/2012

if isempty(x)||isempty(y); k=[]; return; end

%默认特征向量是行
if size(x,2)~=size(y,2); error('The features (number of columns) of X and Y must match!'); end

if nargin<3;  % default: RBF
    type='rbf';
else type=varargin{1}; %“Variable length input argument list"缩写，使用了“可变参数列表机制”的函数允许调用者调用该函数时根据需要来改变输入参数的个数。
end

if strcmpi(type,'rbf')  % RBF  %compare strings
    if nargin<4; sigma=1;   % default: var=1
    else sigma=varargin{2};
    end
    k=exp(-(sum(x.*x,2)*ones(1,size(y,1))+ones(size(x,1),1)*sum(y.*y,2)'-2*x*y')/2/sigma^2);
elseif strcmpi(type,'linear') % linear
    k=x*y';
elseif strcmpi(type,'poly')  % polynominal
    if nargin<4; n=1;       % default: n=1
    else n=varargin{2};
    end
    k=(x*y').^n;
elseif strcmpi(type,'exp')  % exponential
    if nargin<4; sigma=1;   % default: var=1
    else sigma=varargin{2};
    end
    k=exp(x*y'/2/sigma^2);
elseif strcmpi(type,'tanh') || ...
        strcmpi(type,'sigmoid') % sigmoid （not positive semidefinite）
    if nargin<4; a=1;   % default: a=1
    else a=varargin{2};
    end
    if nargin<5; b=1;   % default: b=1
    else b=varargin{3};
    end
    k=tanh(b+a*x*y');
else error('unsupported kernel function!');
end

