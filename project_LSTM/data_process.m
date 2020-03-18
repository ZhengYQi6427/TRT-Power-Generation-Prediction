function [train_data,test_data,test_final,real_test_y]=data_process(beginer,input_num,trainNum,testNum)
%clear
%clc
%trainNum = 30;       %训练样本数
%beginer = 100;
%input_num=4;       %输入节点数
%cell_num=30;        %中间节点数
%output_num=1;       %输出节点数
%testNum =1;        %验证样本数
%% 定义一个时间间隔重新抽样
%规定均匀采样时间点
%load data1000;
load data2000;
t=datenum(time1);
t_fortest=(t-t(19))*10^5;
m=floor((t(1695)-t(19))/datenum(0,0,0,0,0,5)+1);
for i=1:m
    t_newFortest(i)=t_fortest(19)+(t_fortest(1695)-t_fortest(19))/(m-1)*(i-1);
end
t_newFortest=t_newFortest';
t_new=datestr(t_newFortest.*10^(-5)+t(19).*ones(m,1));
%t=datestr(t);
%分段插值
l=length(t_fortest);
l_new=length(t_newFortest);
j=1;
for i=1:l-2
    t_sub=t_fortest(i:(i+2));
    P_sub=P(i:(i+2));
    Ugas_sub=u(i:(i+2));
    T_sub=t(i:(i+2));
    pressure1_sub=p1(i:(i+2));
    while((j<=m)&&(t_newFortest(j)<=t_fortest(i+2))&&((t_newFortest(j)>=t_fortest(i))))
           P_new(j)=lagrange(t_sub,P_sub,t_newFortest(j));
           Ugas_new(j)=lagrange(t_sub,Ugas_sub,t_newFortest(j));
           T_new(j)=lagrange(t_sub,T_sub,t_newFortest(j));
           pressure1_new(j)=lagrange(t_sub,pressure1_sub,t_newFortest(j));
           j=j+1;
    end
end
%% 原始数据归一化
C(:,1)=Ugas_new';
C(:,2)=pressure1_new';
C(:,3)=T_new';
load D;
C(:,4)=D(:,1);
C(:,5)=D(:,2);
C(:,6)=D(:,3);
C(:,7)=D(:,4);
C(:,8)=P_new';
Data=mapminmax(C',0,1)';
%%
train_data = Data(beginer:(trainNum+beginer-1),1:6)';
test_data = Data(beginer:(trainNum+beginer-1),8)';
test_final = Data((trainNum+beginer):(trainNum+testNum+beginer-1),1:6)';
real_test_y = Data((trainNum+beginer):(trainNum+testNum+beginer-1),8)';