function [PA,PAYout]=SPA(PA,sample,varargin)
% 2023/03/28



if nargin==0 || isempty(PA)
    PA=struct('rho',1,...             �߽��ֵ
        'Gamma',[],...          ������ϵ�����洢Ϊ��������
        'kernelsize',1,...      �˳�
        'n_time',0,...
        'InputX',[],...        ����������������ÿ��Ϊһ������������
        'Labels',[]...         ����������������Ӧ��ǩ����������{��1}
        );
    if nargin==0; return; end
end


% ���û�б�ǩ���򲻸��·�������������������������λ�Ϊʲôһ��Ҫ����ǰ�棿��������
if isnumeric(sample) || (length(sample)==1 && iscell(sample))
    if iscell(sample); xn=sample{1};
    else
        xn=sample;
    end
    PAYout=fkernel(xn,PA.InputX,'rbf',PA.kernelsize)*PA.Gamma;
    return;
end

xn=sample{1}; yn=sample{2};
PA.n_time = PA.n_time + 1;

if PA.n_time == 1
    
    PA.Gamma = [PA.Gamma;yn];
    PA.InputX = [PA.InputX;xn];
    PA.Labels = [PA.Labels;yn];
    
    
end

if PA.n_time > 1
    
    Loss = max(0,PA.rho - yn*fkernel(xn,PA.InputX,'rbf',PA.kernelsize)*PA.Gamma);
    probalility_update = min(Loss,1);
    ZZ = binornd(1,probalility_update);
    
    
    if ZZ == 1
        update = Loss*yn;
        PA.Gamma = [PA.Gamma;update];
        PA.InputX = [PA.InputX;xn];
        PA.Labels = [PA.Labels;yn];
    end
    
end


