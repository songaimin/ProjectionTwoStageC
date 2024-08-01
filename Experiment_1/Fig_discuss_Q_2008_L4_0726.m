% q=16的An要小。

clear all; close all; clc;

load Data_2008_SNR_10_L4
InputDimen = 4;
NumExperim = 2;

filename = ['Q_2008_SNR_10_L4'  '_An_01'];


C_1 = 1; C_2 = 2; C_3 = 4; C_4 = 8; C_5 = 16;




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




Dim_1 = zeros(NumExperim,1);
Dim_2 = zeros(NumExperim,1);
Dim_3 = zeros(NumExperim,1);
Dim_4 = zeros(NumExperim,1);
Dim_5 = zeros(NumExperim,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic

for NNN = 1:NumExperim
    NNN
    
    DK_PP_1 = DK_TwoStage();
    DK_PP_1.q = C_1;
    
    
    
    DK_PP_2 = DK_TwoStage();
    DK_PP_2.q = C_2;

    
    
    DK_PP_3 = DK_TwoStage();
    DK_PP_3.q = C_3;

    
    DK_PP_4 = DK_TwoStage();
    DK_PP_4.q = C_4;

    
    DK_PP_5 = DK_TwoStage();
    DK_PP_5.q = C_5;
    DK_PP_5.Anepsilon = 0.1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    
    
    
    SamplingIndex=1;  %每次开始新数据前都要先设为1，本实验中SamplingIndex最高为61
    BERateAPSM_1_K = zeros(NumSamples,1)';
    BERateAPSM_2_K = zeros(NumSamples,1)';
    BERateAPSM_3_K = zeros(NumSamples,1)';
    BERateAPSM_4_K = zeros(NumSamples,1)';
    BERateAPSM_5_K = zeros(NumSamples,1)';
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    for n=1:TrainingTime
        
        
        if ( n <= ChannelChangeTime )
            Observed = Train_Data_A_ALL(1:InputDimen,n,NNN);
            label = Train_Data_A_ALL(InputDimen+1,n,NNN);
        else
            Observed = Train_Data_B_ALL(1:InputDimen,n-ChannelChangeTime,NNN);
            label = Train_Data_B_ALL(InputDimen+1,n-ChannelChangeTime,NNN);
        end
        
        
        DK_PP_1=DK_TwoStage(DK_PP_1,{Observed',label});
        DK_PP_2=DK_TwoStage(DK_PP_2,{Observed',label});
        DK_PP_3=DK_TwoStage(DK_PP_3,{Observed',label});
        DK_PP_4=DK_TwoStage(DK_PP_4,{Observed',label});
        DK_PP_5=DK_TwoStage(DK_PP_5,{Observed',label});
        
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
            
            
            

            [~,y_DK_PP_1]=DK_TwoStage(DK_PP_1,TestingFeatures');
            [~,y_DK_PP_2]=DK_TwoStage(DK_PP_2,TestingFeatures');
            [~,y_DK_PP_3]=DK_TwoStage(DK_PP_3,TestingFeatures');
            [~,y_DK_PP_4]=DK_TwoStage(DK_PP_4,TestingFeatures');
            [~,y_DK_PP_5]=DK_TwoStage(DK_PP_5,TestingFeatures');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            

            BERateAPSM_1_K(SamplingIndex)=sum((TestingLabels.*y_DK_PP_1)<0)/length(y_DK_PP_1);
            BERateAPSM_2_K(SamplingIndex)=sum((TestingLabels.*y_DK_PP_2)<0)/length(y_DK_PP_2);
            BERateAPSM_3_K(SamplingIndex)=sum((TestingLabels.*y_DK_PP_3)<0)/length(y_DK_PP_3);
            BERateAPSM_4_K(SamplingIndex)=sum((TestingLabels.*y_DK_PP_4)<0)/length(y_DK_PP_4);
            BERateAPSM_5_K(SamplingIndex)=sum((TestingLabels.*y_DK_PP_5)<0)/length(y_DK_PP_5);
            
            
            SamplingIndex=SamplingIndex+1;
            
            
        end
        
        
    end
    
    BERateAPSM_1_Kav=BERateAPSM_1_Kav*(NNN-1)/NNN+BERateAPSM_1_K'/NNN;
    BERateAPSM_2_Kav=BERateAPSM_2_Kav*(NNN-1)/NNN+BERateAPSM_2_K'/NNN;
    BERateAPSM_3_Kav=BERateAPSM_3_Kav*(NNN-1)/NNN+BERateAPSM_3_K'/NNN;
    
    BERateAPSM_4_Kav=BERateAPSM_4_Kav*(NNN-1)/NNN+BERateAPSM_4_K'/NNN;
    BERateAPSM_5_Kav=BERateAPSM_5_Kav*(NNN-1)/NNN+BERateAPSM_5_K'/NNN;
    
    
    Dim_1(NNN) = length(DK_PP_1.Labels);
    Dim_2(NNN) = length(DK_PP_2.Labels);
    Dim_3(NNN) = length(DK_PP_3.Labels);
    Dim_4(NNN) = length(DK_PP_4.Labels);
    Dim_5(NNN) = length(DK_PP_5.Labels);
end

Dim1 = mean(Dim_1);
Dim2 = mean(Dim_2);
Dim3 = mean(Dim_3);
Dim4 = mean(Dim_4);
Dim5 = mean(Dim_5);



ChangeBefore = ChannelChangeTime/SamplingInterval;
mean_BER_1_B = sum(BERateAPSM_1_Kav(1:ChangeBefore));
mean_BER_1_A = sum(BERateAPSM_1_Kav(ChangeBefore+1:end));
mean_BER_2_B = sum(BERateAPSM_2_Kav(1:ChangeBefore));
mean_BER_2_A = sum(BERateAPSM_2_Kav(ChangeBefore+1:end));
mean_BER_3_B = sum(BERateAPSM_3_Kav(1:ChangeBefore));
mean_BER_3_A = sum(BERateAPSM_3_Kav(ChangeBefore+1:end));

mean_BER_4_B = sum(BERateAPSM_4_Kav(1:ChangeBefore));
mean_BER_4_A = sum(BERateAPSM_4_Kav(ChangeBefore+1:end));
mean_BER_5_B = sum(BERateAPSM_5_Kav(1:ChangeBefore));
mean_BER_5_A = sum(BERateAPSM_5_Kav(ChangeBefore+1:end));




%%
plot(Index,BERateAPSM_1_Kav,'-<',Index,BERateAPSM_2_Kav,'-d',Index,BERateAPSM_3_Kav,'-v',Index,BERateAPSM_4_Kav,'-p',...
    Index,BERateAPSM_5_Kav,'-s','Linewidth',1.5,'MarkerSize',6)


h = legend([  'q=1 '   ','     ' |SV|= ' num2str(Dim1)], ...
    [ 'q=2 '           ','     ' |SV|= ' num2str(Dim2)], ...
    [  'q=4 '          ','     ' |SV|= ' num2str(Dim3)], ...
    [  'q=8 '          ','     ' |SV|= '  num2str(Dim4)], ...
    [  'q=16 '         ','     ' |SV|= '  num2str(Dim5)]);

grid
xlabel('NUMBER OF TRAINING','Fontsize',12)
ylabel('Misclassification Rate','Fontsize',12)




Fig_Save_Path = 'D:\程序\APSM\分类算法测试结果\';

file_Path_name_fig = [Fig_Save_Path '\' filename '.fig'];
saveas(gcf,file_Path_name_fig);
toc
