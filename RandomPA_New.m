function [PerClass,PerYout]=RandomPA_New(PerClass,sample,varargin)
% 20200215,Ԥ����ȷ�����ֵ�
% ��ԭʼPerceptron_1���޸ģ����ȥ��һ�����������µġ�
% 20200212�޸ģ�ȥ��ƫ����b; ȥ������eat��ȥ��������
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
                'alpha',[],...          ������ϵ�����洢Ϊ��������
                'L',200,...             Budget����
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








    
    


