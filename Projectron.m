function [PP,yOut]=Projectron(PP,sample,varargin)

%perceptron+projectron�ĺ����ļ�

%��ʼֵ�趨���������Ϊ�գ���ʹ������Ĭ��ֵ��
if nargin==0 || isempty(PP)
    PP=struct('Gamma',[],...        
        'InputX',[],...      
        'Labels',[],...
        'eta_error',0.001,...
        'kernelsize',3.2,...
        'n_time',0,...
        'K_inv',[]...
        );
    if nargin==0; return; end
end


% ���û�б�ǩ���򲻸��·���������������������
if isnumeric(sample) || (length(sample)==1 && iscell(sample))
    if iscell(sample); xn=sample{1}; 
    else
        xn = sample;
    end
    yOut = fkernel(xn,PP.InputX,'rbf',PP.kernelsize)*PP.Gamma; 
    return;
end

xn = sample{1}; yn = sample{2};


PP.n_time = PP.n_time + 1;

if PP.n_time == 1
    PP.Gamma=[PP.Gamma;yn];
    PP.InputX=[PP.InputX;xn];
    PP.Labels=[PP.Labels;yn];
    PP.K_inv=1/fkernel(xn,xn,'rbf',PP.kernelsize);%�������
end

if PP.n_time > 1
    y_predict=sign(yn*fkernel(xn,PP.InputX,'rbf',PP.kernelsize)*PP.Gamma);
    
    if y_predict < 0 %����
        k_n=fkernel(xn,PP.InputX,'rbf',PP.kernelsize);
        coeff_Proj=PP.K_inv*(k_n');
        Proj_error=fkernel(xn,xn,'rbf',PP.kernelsize)-k_n*coeff_Proj;
        if sqrt(Proj_error) < PP.eta_error
            PP.Gamma=PP.Gamma+yn*coeff_Proj;
        else
            PP.Gamma=[PP.Gamma;yn];
            PP.InputX=[PP.InputX;xn];
            PP.Labels=[PP.Labels;yn];
            K_inv_temp=zeros(length(PP.Gamma));
            K_inv_temp(1:(end-1),1:(end-1))=PP.K_inv;
            colu_coeff = [coeff_Proj;-1];
            PP.K_inv = K_inv_temp +(1/Proj_error)*(colu_coeff*colu_coeff');
        end
    end
end


