function [NORClass,NORYout]=NORMAClassifierNoVTrick(NORClass,sample,varargin)
%      NORMA,��ʹ��V-trick
%      NORClass.f            ��������ֵ��Ĭ��0.
%      NORClass.rho          margin�ĳ�ֵ��Ĭ��0.
%              .eta          ����etaǰ���ϵ��
%              .b            ������ƫ�������������
%              .reg          ����������ȡֵΪ0��������ȡֵΪ1��Ĭ��0.
%              .alpha        ������ϵ��
%              .samples      ��������������
%              .labels       ����������������Ӧ��ǩ��{��1}
%              .kernelfun    �˺��������Ĭ���Ƿ���Ϊ1�ĸ�˹�ˡ�
%                            ������ʽӦΪ��y=kernelfun(x1,x2)��
%   sample     �������������Ϊ{x,y}�Ͷ�ԪԪ�����飬��xΪ��������(������)��
%              yΪ��ǩ����1����ͬʱ�����ݽ����ڸ��·����������Ϊ��һ
%              ��������x��{x}����������������������������ݣ�ȱ�ٱ�ǩ����ʱ
%              ���������������������·�������
%   �����
%         y   �����������ķ��������������˿ռ��������㵽��ƽ��ľ��룩��
%         CLASSIFIER   ���º�ķ����������������ѵ����������
%
%   e.g.
%            NORClass=kernelapsm([],{x(1,:),1});        % ��1��ѵ������
%            NORClass=kernelapsm(NORClass,{x(2,:),-1}); % ��2��ѵ������
%            ...
%

% Ref:Online learning with kernels,2004,IEEE SP

% by SAM
% 2019/02/20

%��������ʼֵ�趨�����������գ���ʹ������Ĭ��ֵ��
if nargin==0 || isempty(NORClass)
        NORClass=struct('f',0,...       �ó���û���õ�f����¼���Ǳ������
                'rho',0,...             �߽��ֵ
                'Coeffeta',1,...        etaϵ����ͨ��eta=Coeffeta/sqrt(n)
                'b',0,...               ������ƫ��������ǰ����ֵ��
                'alpha',[],...          ������ϵ�����洢Ϊ��������
                'sigma',1,...           ���ڳ�ʼֵ��Ĭ�Ϸ�����󣬷�����Щ������ʼ�Ͳ����£�
                'reg',1,...             ���򻯲�����0��ʾ��ʹ�����򻯣�1��ʾ����ϵ��lambdaΪ1��
                'kernelsize',2,...      �˳�
                'samples',[],...        ����������������ÿ��Ϊһ������������
                'labels',[]...         ����������������Ӧ��ǩ����������{��1}
                );
        if nargin==0; return; end
end


% ���û�б�ǩ���򲻸��·�������������������������λ�Ϊʲôһ��Ҫ����ǰ�棿��������
if isnumeric(sample) || (length(sample)==1 && iscell(sample))
        if iscell(sample); xn=sample{1};
        else xn=sample;
        end
        NORYout=fkernel(xn,NORClass.samples,'rbf',NORClass.kernelsize)*NORClass.alpha + NORClass.b;
        return;
end

xn=sample{1}; yn=sample{2};
NORClass.samples = [NORClass.samples;xn];  % ÿ��������һ���С�
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








    
    


