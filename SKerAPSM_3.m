function [SKer,yout]=SKerAPSM_3(SKer,sample,varargin)

% �̶�������mu = 1;
% ԭ���ĳ���������������ˣ���Ϊ��������
%         SKer.rho=max(0,SKer.nu*(SKer.theta-...
%             -SKer.theta0)+SKer.rho0);  
% ��ȷ����
%         SKer.rho=max(0,SKer.nu*(SKer.theta-...
%             SKer.theta0)+SKer.rho0); 


if nargin==0 || isempty(SKer)
    SKer=struct('rho0',1,...
        'rho',1,...             �߽�ֵ����ֵ�뵱ǰ����ֵ����rho0Ĭ��1.
        'theta0',1,...
        'theta',1,...           �߽�ֵ��������ģ���Ա�����theta0Ĭ��1.
        'dtheta',1e-3,...       theta�ı仯������Ĭ��1e03.
        'nu',0.1,...            �߽�ֵ��������ģ��б��.Ĭ��0.1.
        'R',0.5,...             ���жȣ����ڱ߽�ֵ��������Ĭ��0.5.
        'q',4,...               ͬʱ�������������Ĭ��16.
        'b',0,...               ������ƫ��������ǰ����ֵ��
        'muAdj',0.1,...           ͶӰ�������ϵ����mu=muAdj*Mn�������Ϊ1�������ʱ����.
        'mu',0,...              ͶӰ�������.Ĭ��1.��muAdj=0ʱ��Ч��
        'fnorm',0,...           ��ǰ�������Ķ�����
        'delta',5,...           ͶӰ����뾶��Ĭ��1.
        'L',700,...             ϡ�軯�����������������Ĭ��100.
        'gamma',[],...          ������ϵ������������
        'InputX',[],...         ����������������ÿ��Ϊһ������������
        'labels',[],...         ����������������Ӧ��ǩ����������{��1}
        'kernelfun',@RBFKernel...   �˺��������Ĭ���Ƿ���Ϊ1�ĸ�˹�ˡ�
        );
    if nargin==0; return; end   % ���ǣ���������
end

% ����ѡ��
bexRho=false; % �Ƿ���б߽�ֵ�ĵ�����true��������������ʹ��classifier�е�rho���㡣
bnoSparse=false; % �Ƿ����ϡ�軯��true����������ϡ�軯������ɾ���������������ϡ�軯��
bnoBall=false; % �Ƿ�ͶӰ������true������ͶӰ������ͶӰ������
if nargin>2
    for n=1:length(varargin)
        switch lower(varargin{n})   % ���ǣ���������
            case 'exrho'; bexRho=true;
            case 'nosparse'; bnoSparse=true;
            case 'bnoball'; bnoBall=true;
        end
    end
end


% ���û�б�ǩ���򲻸��·���������������������
if isnumeric(sample) || (length(sample)==1 && iscell(sample)) % ???????
    if iscell(sample); xn=sample{1}; %Ϊʲô��{}��������
    else xn=sample;
    end
    yout=EvalOutput(SKer,xn);
    return;  %%Ϊʲreturn???
end


xn=sample{1}; yn=sample{2};

%����ϵ��beta.
% ����������sample, ������samples, ���߲�һ����
n_samples=size(SKer.InputX,1)+1; % ��ǰ��������    %Ϊʲô����length����size��ΪʲôҪ+1��


if n_samples < SKer.q
    Jn=1:n_samples;
    SKer.w = ones(n_samples,1)/n_samples;
else
    Jn=( n_samples-SKer.q+1 ):n_samples;      % ���SKer.q����������
    SKer.w=ones(SKer.q,1)/SKer.q;
end


x=[SKer.InputX(Jn(1:end-1),:);xn];  % ���q������
y=[SKer.labels(Jn(1:end-1));yn];
Gn=SKer.kernelfun(x,x); % ���q��������gram����
anorm=1+diag(Gn); % ������aj�ķ�����1+k(xj,xj)(������)
if n_samples==1; 
    g=SKer.b; % �������ĳ�ֵΪ�㣬��˵�һ�����ڸ�ά�ռ��ڻ���f(x)ҲΪ0.
else
    g=EvalOutput(SKer,x); %���q�������ķ��������
end

beta=max(0,SKer.rho-y.*g).*(SKer.w).*y./anorm; % ���ڷ��������µı���beta��������


if SKer.muAdj~=0; % ʹ�ö�̬������muֵ
    if SKer.muAdj<0 || SKer.muAdj>2
        error('the parameter ''SKer.muAdj'' must be within the interval, [0,2].');
    end
    An = anorm'*(beta.^2./SKer.w);  %һ�г�һ�У�An����
    if An==0; % ��ǰ���������ڸ���͹���Ľ�����
        SKer.mu=0; %��1���0����ʵ��An==0ʱ��Mn=0������SKer.mu=0������ֱ����SKer.mu=0
    else
%         Bn=beta'*(1+Gn)*beta;
%         Mn=An/Bn;
%         SKer.mu=SKer.muAdj*Mn;  
        SKer.mu = 1;
    end
end



% ���·�����
SKer.gamma(n_samples,1)=0; %�������gamma^(n_samples)_(n_samples)=0;
SKer.gamma(Jn)=SKer.gamma(Jn)+SKer.mu*beta;
fnorm=SKer.fnorm+2*SKer.mu*beta'...
    *(g-SKer.b)+SKer.mu^2*beta'*Gn*beta; % ��������������ά�ռ�ĳ��ȣ���


%��ͶӰ����ϡ��
if ~bnoBall % ͶӰ������
    sfnorm=sqrt(fnorm);
    if sfnorm>SKer.delta
        SKer.gamma=SKer.gamma*SKer.delta/sfnorm;
        fnorm=SKer.delta^2;
    end
end




if ~bnoSparse % ����ϡ�軯
    if n_samples>SKer.L %�ﵽ�������Ƴ�����������㣬���޸ķ�����������
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



if ~bexRho; % ���±߽�ֵ
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
