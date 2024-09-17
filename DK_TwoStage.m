function [MultiKer,youtMulti]=DK_TwoStage(MultiKer,sample,varargin)



if nargin==0 || isempty(MultiKer)
    MultiKer = struct('q',4,...
        'muAdj',0.2,...
        'L',200,...
        'DimMaxSub',1,...
        'Sigma1',0.5,...
        'Sigma2',4,...
        'CohThr',0.1,...
        'Anepsilon',0.3,...
        'muNLMS',0.2,...
        'rho',1,...
        'InputX',[],...
        'Labels',[],...
        'gamma1',[],...
        'gamma2',[]...
        );
    if nargin==0; return; end   % ���ǣ���������
end


% ���û�б�ǩ���򲻸��·���������������������
if isnumeric(sample) || (length(sample)==1 && iscell(sample)) % ???????
    if iscell(sample); xn=sample{1}; %Ϊʲô��{}��������
    else xn=sample;
    end
    youtMulti= MultiEvalOutput(MultiKer,xn);
    return;  %%Ϊʲreturn???
end


xn=sample{1}; yn=sample{2};



n_InputX=size(MultiKer.InputX,1)+1; % ��ǰ��������    %Ϊʲô����length����size��ΪʲôҪ+1��

if n_InputX==1;
    Error_bound = 0;
else
    Error_bound = MultiKer.Anepsilon;
end


if  n_InputX <= MultiKer.L
   
    if n_InputX < MultiKer.q
        Jn=1:n_InputX;
    else
        Jn=( n_InputX-MultiKer.q+1 ):n_InputX;      % ���SKer.q����������
    end
    
    
    
    x=[MultiKer.InputX(Jn(1:end-1),:);xn];  % ���q������
    y=[MultiKer.Labels(Jn(1:end-1));yn];
    
    
    if n_InputX==1;
        Multi_g=0; % �������ĳ�ֵΪ�㣬��˵�һ�����ڸ�ά�ռ��ڻ���f(x)ҲΪ0.
    else
        Multi_g=MultiEvalOutput(MultiKer,x); %���q�������ķ��������
    end
    

    
    anorm = 2; %˫��
    
    error_beta = max(0,MultiKer.rho - y.*Multi_g);
    nonzers_error = nnz(error_beta); %����з���ĸ���
    if nonzers_error == 0
        w_projection = 0;    %��ʱ�൱�ڲ�����
    else
        w_projection = 1/nonzers_error;
    end
    
    AnErrorSquare = (w_projection) * (error_beta'*error_beta);
    beta = w_projection*(error_beta.*y)/anorm;
    An = AnErrorSquare/anorm;
    
    if AnErrorSquare >= Error_bound
        Gram_1 = fkernel(x,x,'rbf',MultiKer.Sigma1);
        Gram_2 = fkernel(x,x,'rbf',MultiKer.Sigma2);
        Bn = beta'*(Gram_1 + Gram_2)*beta;
        Mn = An/Bn;
        mu = MultiKer.muAdj*Mn;
        
        MultiKer.gamma1(n_InputX,1) = 0;
        MultiKer.gamma2(n_InputX,1) = 0;
        
        coefficient_update = mu*beta;
        MultiKer.gamma1(Jn)=MultiKer.gamma1(Jn) + coefficient_update;
        MultiKer.gamma2(Jn)=MultiKer.gamma2(Jn) + coefficient_update;
        MultiKer.InputX(end+1,:) = xn; %̫Ư���ˣ���������ô�á�(����end�����)
        MultiKer.Labels(end+1,1) = yn;  %̫Ư���ˣ���������ô�á�
    end
end


%%


if n_InputX > MultiKer.L
    Multi_g=MultiEvalOutput(MultiKer,xn); %���q�������ķ��������
    error_beta = max(0,MultiKer.rho - yn*Multi_g);
    AnErrorSquare = error_beta^(2);
    beta = error_beta*yn;
    
    
    if AnErrorSquare >= Error_bound
        CoherenceXnDiction_1 = fkernel(xn,MultiKer.InputX,'rbf',MultiKer.Sigma1);
        %CoherenceXnDiction_2 = fkernel(xn,MultiKer.InputX,'rbf',MultiKer.Sigma2);
        
        MaxCoherence = max(CoherenceXnDiction_1);
        %MaxCoherence = max(CoherenceXnDiction_2);
        
        if  MaxCoherence >= MultiKer.CohThr
            
            [~,MaxPosition] = sort(CoherenceXnDiction_1,2,'descend');
            %[~,MaxPosition] = sort(CoherenceXnDiction_2,2,'descend');
            
            IndexCoherence = MaxPosition(1:MultiKer.DimMaxSub);
            
            
            
            SubDictionary = MultiKer.InputX(IndexCoherence,:); %�ֵ�������ͬһ���������ӿռ乲��
            
            GramSubDictionary_1 = fkernel(SubDictionary,SubDictionary,'rbf',MultiKer.Sigma1);
            CrossCorrelation_1 = CoherenceXnDiction_1(IndexCoherence);
            
            AlphaProjection_1 = GramSubDictionary_1\CrossCorrelation_1';
            betaXn_1 = beta(end)/(CrossCorrelation_1*AlphaProjection_1);
            
            gammaProjection_1 = betaXn_1*AlphaProjection_1;
            
            MultiKer.gamma1(IndexCoherence) = MultiKer.gamma1(IndexCoherence) + MultiKer.muNLMS*gammaProjection_1;

            
        else
            
            MultiKer.gamma1(n_InputX,1) = 0;
            MultiKer.gamma2(n_InputX,1) = 0;
            MultiKer.gamma1(end)=MultiKer.gamma1(end) + MultiKer.muNLMS*beta;
 
            MultiKer.InputX(end+1,:) = xn; %̫Ư���ˣ���������ô�á�(����end�����)
            MultiKer.Labels(end+1,1) = yn;  %̫Ư���ˣ���������ô�á�
        end
    end
    
end







function g=MultiEvalOutput(MultiKer,x)
g=fkernel(x,MultiKer.InputX,'rbf',MultiKer.Sigma1)*MultiKer.gamma1 + ...
    fkernel(x,MultiKer.InputX,'rbf',MultiKer.Sigma2)*MultiKer.gamma2;


