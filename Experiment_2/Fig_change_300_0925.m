


clear all; close all; clc;
load Data_New_NLC_1_SNR_20_L8_change300

InputDimen = 8;
NumExperim = 50;

filename = ['Fig_change300'  '_3'];


rho_NORMA = 0;  PAKerSize_NORMA = 5;  Coeffeta = 0.001;

rho_RPA = 0; PAKerSize_RPA = 4; Budget_RPA = 200;

rho_F = 0;  PAKerSize_F = 4;  Budget_F = 200;

q_S_1 = 16; delta_S_1 = 5;  L_S_1 = 500;  SK_APSM = 4;

MKerSize_2 = 4;

%%%%%%%%%%%%%%%%%%%%%%%%%
Train_A = size(Train_Data_A_ALL);
Train_B = size(Train_Data_B_ALL);
%训练数据是A和B的和
TrainingTime = Train_A(2) + Train_B(2);
ChannelChangeTime = Train_A(2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%这个主要是用于画图，每隔25点画一次。如果每一点都画，太密。
SamplingInterval = 25;
NumSamples=TrainingTime/SamplingInterval;
Index=SamplingInterval*(1:NumSamples)';
BERateAPSM_1_Kav=zeros(NumSamples,1);
BERateAPSM_2_Kav=zeros(NumSamples,1);
BERateAPSM_3_Kav=zeros(NumSamples,1);
BERateAPSM_4_Kav=zeros(NumSamples,1);
BERateAPSM_5_Kav=zeros(NumSamples,1);

BERateAPSM_6_Kav=zeros(NumSamples,1);
BERateAPSM_7_Kav=zeros(NumSamples,1);
BERateAPSM_8_Kav=zeros(NumSamples,1);
BERateAPSM_9_Kav=zeros(NumSamples,1);
BERateAPSM_10_Kav=zeros(NumSamples,1);




Dim_4 = zeros(NumExperim,1);
Dim_5 = zeros(NumExperim,1);
Dim_6 = zeros(NumExperim,1);
Dim_7 = zeros(NumExperim,1);
Dim_8 = zeros(NumExperim,1);
Dim_10 = zeros(NumExperim,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic

for NNN = 1:NumExperim
    NNN
    
    
    PA_1 = NORMAClassifierNoVTrick();
    PA_1.Coeffeta = Coeffeta;
    PA_1.kernelsize = PAKerSize_NORMA;
    PA_1.rho = rho_NORMA;
    
    
    PA_2 = RandomPA_New();
    PA_2.rho = rho_RPA;
    PA_2.kernelsize = PAKerSize_RPA;
    PA_2.L = Budget_RPA;
    
    
    PA_3 = Forgetron();
    PA_3.rho = rho_F;
    PA_3.kernelsize = PAKerSize_F;
    PA_3.Budget = Budget_F;
    
    PA_4 = Projectron();
    PA_4.kernelsize = 1;
    PA_4.eta_error = 0.001;
    
    PA_5 = Projectron_Plus();
    PA_5.kernelsize = 1;
    PA_5.eta_error = 0.001;
    
    
    PA_6 = OLRU();
    PA_6.kernelsize = 2;

    PA_7 = OLRD();
    PA_7.kernelsize = 2;
    
    PA_8 = SPA();
    PA_8.kernelsize = 2;
    
    
    
    SKer3=SKerAPSM_3();
    SKer3.q = q_S_1;
    SKer3.delta = delta_S_1;
    SKer3.L = L_S_1;
    SKer3.kernelfun=@(x1,x2)   fkernel(x1,x2,'rbf',SK_APSM);
    
    
    
    DK_PP_1 = DK_TwoStage();
    DK_PP_1.Sigma2 = MKerSize_2;

    
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    
    
    SamplingIndex=1;  %每次开始新数据前都要先设为1，本实验中SamplingIndex最高为61
    BERateAPSM_1_K = zeros(NumSamples,1)';
    BERateAPSM_2_K = zeros(NumSamples,1)';
    BERateAPSM_3_K = zeros(NumSamples,1)';
    BERateAPSM_4_K = zeros(NumSamples,1)';
    BERateAPSM_5_K = zeros(NumSamples,1)';
    
    BERateAPSM_6_K = zeros(NumSamples,1)';
    BERateAPSM_7_K = zeros(NumSamples,1)';
    BERateAPSM_8_K = zeros(NumSamples,1)';
    BERateAPSM_9_K = zeros(NumSamples,1)';
    BERateAPSM_10_K = zeros(NumSamples,1)';
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    for n=1:TrainingTime
        
        
        if ( n <= ChannelChangeTime )
            Observed = Train_Data_A_ALL(1:InputDimen,n,NNN);
            label = Train_Data_A_ALL(InputDimen+1,n,NNN);
        else
            Observed = Train_Data_B_ALL(1:InputDimen,n-ChannelChangeTime,NNN);
            label = Train_Data_B_ALL(InputDimen+1,n-ChannelChangeTime,NNN);
        end
        
        
        
        PA_1 = NORMAClassifierNoVTrick(PA_1,{Observed',label});
        PA_2 = RandomPA_New(PA_2,{Observed',label});
        PA_3 = Forgetron(PA_3,{Observed',label});
        PA_4 = Projectron(PA_4,{Observed',label});
        PA_5 = Projectron_Plus(PA_5,{Observed',label});
        PA_6 = OLRU(PA_6,{Observed',label});
        PA_7 = OLRD(PA_7,{Observed',label});
        PA_8 = SPA(PA_8,{Observed',label});
        
        
        SKer3=SKerAPSM_3(SKer3,{Observed',label});
        DK_PP_1=DK_TwoStage(DK_PP_1,{Observed',label});

        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% Test the classifiers. %%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        SamplingFlag=mod(n,SamplingInterval);
        
        if ( SamplingFlag == 0 )   %如果是SamplingInterval的倍数，则计算分类的正确率，否则不计算。
            
            if ( n <= ChannelChangeTime )
                TestingFeatures = Test_Data_A_ALL(1:InputDimen,:,NNN);
                TestingLabels = Test_Data_A_ALL(InputDimen+1,:,NNN)';
            else
                TestingFeatures = Test_Data_B_ALL(1:InputDimen,:,NNN);
                TestingLabels = Test_Data_B_ALL(InputDimen+1,:,NNN)';
            end
            
            
            
            [~,y_PA_1] = NORMAClassifierNoVTrick(PA_1,TestingFeatures');
            
            [~,y_PA_2] = RandomPA_New(PA_2,TestingFeatures');
            
            [~,y_PA_3] = Forgetron(PA_3,TestingFeatures');
            
            [~,y_PA_4] = Projectron(PA_4,TestingFeatures');
            
            [~,y_PA_5] = Projectron_Plus(PA_5,TestingFeatures');
            
            [~,y_PA_6] = OLRU(PA_6,TestingFeatures');
            
            [~,y_PA_7] = OLRD(PA_7,TestingFeatures');
            
            [~,y_PA_8] = SPA(PA_8,TestingFeatures');
                       
            [~,y_SKer3]=SKerAPSM_3(SKer3,TestingFeatures');
            
            [~,y_DK_PP_1]=DK_TwoStage(DK_PP_1,TestingFeatures');
            
            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            BERateAPSM_1_K(SamplingIndex)=sum((TestingLabels.*y_PA_1)<0)/length(y_PA_1);
            BERateAPSM_2_K(SamplingIndex)=sum((TestingLabels.*y_PA_2)<0)/length(y_PA_2);
            BERateAPSM_3_K(SamplingIndex)=sum((TestingLabels.*y_PA_3)<0)/length(y_PA_3);
            BERateAPSM_4_K(SamplingIndex)=sum((TestingLabels.*y_PA_4)<0)/length(y_PA_4);
            BERateAPSM_5_K(SamplingIndex)=sum((TestingLabels.*y_PA_5)<0)/length(y_PA_5);
            BERateAPSM_6_K(SamplingIndex)=sum((TestingLabels.*y_PA_6)<0)/length(y_PA_6);
            BERateAPSM_7_K(SamplingIndex)=sum((TestingLabels.*y_PA_7)<0)/length(y_PA_7);
            BERateAPSM_8_K(SamplingIndex)=sum((TestingLabels.*y_PA_8)<0)/length(y_PA_8);
            BERateAPSM_9_K(SamplingIndex)=sum((TestingLabels.*y_SKer3)<0)/length(y_SKer3);
            BERateAPSM_10_K(SamplingIndex)=sum((TestingLabels.*y_DK_PP_1)<0)/length(y_DK_PP_1);
            
            
            
            
            SamplingIndex=SamplingIndex+1;
            
            
        end
        
        
    end
    
    BERateAPSM_1_Kav=BERateAPSM_1_Kav*(NNN-1)/NNN+BERateAPSM_1_K'/NNN;
    BERateAPSM_2_Kav=BERateAPSM_2_Kav*(NNN-1)/NNN+BERateAPSM_2_K'/NNN;
    BERateAPSM_3_Kav=BERateAPSM_3_Kav*(NNN-1)/NNN+BERateAPSM_3_K'/NNN;
    BERateAPSM_4_Kav=BERateAPSM_4_Kav*(NNN-1)/NNN+BERateAPSM_4_K'/NNN;
    BERateAPSM_5_Kav=BERateAPSM_5_Kav*(NNN-1)/NNN+BERateAPSM_5_K'/NNN;
    
    BERateAPSM_6_Kav=BERateAPSM_6_Kav*(NNN-1)/NNN+BERateAPSM_6_K'/NNN;
    BERateAPSM_7_Kav=BERateAPSM_7_Kav*(NNN-1)/NNN+BERateAPSM_7_K'/NNN;
    BERateAPSM_8_Kav=BERateAPSM_8_Kav*(NNN-1)/NNN+BERateAPSM_8_K'/NNN;
    BERateAPSM_9_Kav=BERateAPSM_9_Kav*(NNN-1)/NNN+BERateAPSM_9_K'/NNN;
    BERateAPSM_10_Kav=BERateAPSM_10_Kav*(NNN-1)/NNN+BERateAPSM_10_K'/NNN;
    

    Dim_4(NNN) = length(PA_4.Labels);
    Dim_5(NNN) = length(PA_5.Labels);
    Dim_6(NNN) = length(PA_6.Labels);
    Dim_7(NNN) = length(PA_7.Labels);
    Dim_8(NNN) = length(PA_8.Labels);
    Dim_10(NNN) = length(DK_PP_1.Labels);
end



Dim4 = mean(Dim_4);
Dim5 = mean(Dim_5);
Dim6 = mean(Dim_6);
Dim7 = mean(Dim_7);
Dim8 = mean(Dim_8);
Dim10 = mean(Dim_10);






%%
plot(Index,BERateAPSM_1_Kav,'b-.',Index,BERateAPSM_3_Kav,'y-o',Index,BERateAPSM_5_Kav,'m:+',...
    Index,BERateAPSM_6_Kav,'r-*',Index,BERateAPSM_7_Kav,'g-v',Index,BERateAPSM_8_Kav,'r:s',...
    Index,BERateAPSM_9_Kav,'k-h',Index,BERateAPSM_10_Kav,'b-d','Linewidth',2.0,'MarkerSize',8)

set(gca, 'FontSize', 16);


h = legend([ 'NORMA'  ], ...
    [ 'Forgetron'  ', '   ' |SV|=' num2str(Budget_F)], ...
    [ 'Projectron++'  ', '   ' |SV|='    num2str(Dim5)], ...
    [ 'OLRU'  ', '   ' |SV|='    num2str(Dim6)], ...
    [ 'OLRD'  ', '   ' |SV|='    num2str(Dim7)], ...
    [ 'SPA'  ', '   ' |SV|='    num2str(Dim8)], ...
    [ 'APSM'       ', '   ' |SV|=' num2str(L_S_1)], ...
    [ 'Proposed'   ', '   ' |SV|=' num2str(Dim10)]);

set(h,'FontSize',18)

grid
xlabel('NUMBER OF TRAINING','Fontsize',16)
ylabel('Misclassification Rate','Fontsize',16)



% Fig_Save_Path = 'D:\程序\APSM\分类算法测试结果\';
% 
% file_Path_name_fig = [Fig_Save_Path '\' filename '.fig'];
% saveas(gcf,file_Path_name_fig);

print(gcf, 'my_plot3.png', '-dpng', '-r300'); % '-r300' 设置分辨率为300 dpi
toc
