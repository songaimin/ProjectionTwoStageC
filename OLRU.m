function [OL,PAYout]=OLRU(OL,sample,varargin)
% 2023/03/28
% PAYout=fkernel(xn',OL.InputX,'rbf',OL.kernelsize)*OL.Gamma;

%��������ʼֵ�趨�����������գ���ʹ������Ĭ��ֵ��
if nargin==0 || isempty(OL)
    OL=struct( 'rho',1,...             �߽��ֵ
        'Gamma',[],...          ������ϵ�����洢Ϊ��������
        'kernelsize',4.1,...      �˳�
        'n_time',0,...
        'c',1.2,...
        'alpha',0.2,...
        'eta',0.7,...
        'TrainingTime',1500,...
        'InputX',[],...        ����������������ÿ��Ϊһ������������
        'Labels',[]...         ����������������Ӧ��ǩ����������{��1}
        );
    if nargin==0; return; end
end


% ���û�б�ǩ���򲻸��·�������������������������λ�Ϊʲôһ��Ҫ����ǰ�棿��������
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
    
    if Loss > 0 && theta == 0   %����
        
        update = OL.eta*Loss*yn;
        OL.Gamma = [OL.Gamma; update];
        OL.InputX = [OL.InputX;xn];
        OL.Labels = [OL.Labels;yn];
    end
    
end


