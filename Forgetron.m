function [PerClass,PerYout]=Forgetron(PerClass,sample,varargin)
% ����FixData_Forgetron_3��д�ɺ�����
%      PerClass.f            ��������ֵ��Ĭ��0.
%      PerClass.rho          margin�ĳ�ֵ��Ĭ��0.
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
%            PerClass=kernelapsm([],{x(1,:),1});        % ��1��ѵ������
%            PerClass=kernelapsm(PerClass,{x(2,:),-1}); % ��2��ѵ������
%            ...
%

% Ref:Online learning with kernels,2004,IEEE SP

% by SAM
% 2019/02/20

%��������ʼֵ�趨�����������գ���ʹ������Ĭ��ֵ��
if nargin==0 || isempty(PerClass)
        PerClass=struct('f',0,...       �ó���û���õ�f����¼���Ǳ������
                'rho',0,...             �߽��ֵ
                'NMistake',0,...        ������
                'Q',0,...               Q_Psi
                'ShrOld',1,...          ���ϵ������ϵ��
                'Budget',200,...        Budget
                'alpha',[],...          ������ϵ�����洢Ϊ��������
                'kernelsize',1,...      �˳�
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
        PerYout=fkernel(xn,PerClass.samples,'rbf',PerClass.kernelsize)*PerClass.alpha;
        return;
end

xn=sample{1}; yn=sample{2};


ttt = length(PerClass.alpha) + 1;


if ttt == 1
    PerClass.samples = [PerClass.samples;xn];
    PerClass.labels = [PerClass.labels;yn];
    PerClass.alpha = [PerClass.alpha ; yn];
    PerClass.NMistake = PerClass.NMistake + 1;
end



if ttt > 1
    
    Gn = fkernel(xn,PerClass.samples,'rbf',PerClass.kernelsize)*PerClass.alpha;
    gOut = yn * Gn;
    
    if gOut <= PerClass.rho
        if ttt <= PerClass.Budget
            PerClass.samples = [PerClass.samples;xn];
            PerClass.labels = [PerClass.labels;yn];
            PerClass.alpha = [PerClass.alpha ; yn];
            PerClass.NMistake = PerClass.NMistake + 1;
        else
            PerClass.samples = [PerClass.samples;xn];
            PerClass.labels = [PerClass.labels;yn];
            PerClass.alpha = [PerClass.alpha ; yn];
            PerClass.NMistake = PerClass.NMistake + 1;
            
            gOut_Oldest = fkernel(PerClass.samples(1,:),PerClass.samples,'rbf',PerClass.kernelsize)*PerClass.alpha;
            psi_a = (PerClass.ShrOld)^(2) - 2*PerClass.ShrOld*PerClass.labels(1)*gOut_Oldest;
            psi_b = 2*PerClass.ShrOld;
            psi_c = PerClass.Q - (15/32)*PerClass.NMistake;
            psi_d = (psi_b)^(2) - 4*psi_a*psi_c;
            
            if  psi_a>0    
                shrink_Alpha = min(1,(-psi_b+sqrt(psi_d))/(2*psi_a));
            elseif psi_a<0 && psi_d>0 && ((-psi_b-sqrt(psi_d))/(2*psi_a))>1
                shrink_Alpha = min(1,(-psi_b+sqrt(psi_d))/(2*psi_a));
            elseif psi_a==0
                shrink_Alpha = min(1,-psi_c/psi_b);
            else
                shrink_Alpha = 1;
            end
            
            PerClass.ShrOld = PerClass.ShrOld*shrink_Alpha;
            PerClass.Q = PerClass.Q + (PerClass.ShrOld^(2) + 2*PerClass.ShrOld - 2*PerClass.ShrOld*PerClass.labels(1)*shrink_Alpha*gOut_Oldest);
            
            PerClass.alpha = shrink_Alpha*PerClass.alpha;
            PerClass.alpha(1) = [];
            PerClass.samples(1,:) = [];
            PerClass.labels(1) = [];
        end
    end
    
end








    
    


