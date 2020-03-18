
clear all;
clc;
trainNum = 10;       %训练样本数
Beginer = 500;
for beginer = 1:Beginer
    %% 数据加载，并归一化处理
    % 结点数设置
    input_num=7;       %输入节点数
    cell_num=30;        %中间节点数
    output_num=1;       %输出节点数
    
    testNum =1;        %验证样本数？？
    [train_data,test_data,test_final,real_test_y]=LSTM_data_process(beginer,input_num,trainNum,testNum);
    data_length=size(train_data,1);
    data_num=size(train_data,2);
    %% 网络参数初始化

    % 网络中门的偏置
    bias_input_gate=rand(1,cell_num);
    bias_forget_gate=rand(1,cell_num);
    bias_output_gate=rand(1,cell_num);
    % ab=1.2;
    % bias_input_gate=ones(1,cell_num)/ab;
    % bias_forget_gate=ones(1,cell_num)/ab;
    % bias_output_gate=ones(1,cell_num)/ab;
    %网络权重初始化
    ab=20;
    weight_input_x=rand(input_num,cell_num)/ab;
    weight_input_h=rand(output_num,cell_num)/ab;
    weight_inputgate_x=rand(input_num,cell_num)/ab;
    weight_inputgate_c=rand(cell_num,cell_num)/ab;
    weight_forgetgate_x=rand(input_num,cell_num)/ab;
    weight_forgetgate_c=rand(cell_num,cell_num)/ab;
    weight_outputgate_x=rand(input_num,cell_num)/ab;
    weight_outputgate_c=rand(cell_num,cell_num)/ab;

    %hidden_output权重
    weight_preh_h=rand(cell_num,output_num);

    %网络状态初始化
    cost_gate=1e-6;
    h_state=rand(output_num,data_num);
    cell_state=rand(cell_num,data_num);
    %% 网络训练学习
    for iter=1:3000
        yita=0.01;            %每次迭代权重调整比例
        for m=1:data_num
            %前馈部分
            if(m==1)
                gate=tanh(train_data(:,m)'*weight_input_x);
                input_gate_input=train_data(:,m)'*weight_inputgate_x+bias_input_gate;
                output_gate_input=train_data(:,m)'*weight_outputgate_x+bias_output_gate;
                for n=1:cell_num
                    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
                end
                forget_gate=zeros(1,cell_num);
                forget_gate_input=zeros(1,cell_num);
                cell_state(:,m)=(input_gate.*gate)';
            else
                gate=tanh(train_data(:,m)'*weight_input_x+h_state(:,m-1)'*weight_input_h);
                input_gate_input=train_data(:,m)'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
                forget_gate_input=train_data(:,m)'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
                output_gate_input=train_data(:,m)'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
                for n=1:cell_num
                    input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
                    forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
                    output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
                end
                cell_state(:,m)=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';   
            end
            pre_h_state=tanh(cell_state(:,m)').*output_gate;
            h_state(:,m)=(pre_h_state*weight_preh_h)';
            %误差计算
            Error=h_state(:,m)-test_data(:,m);
            Error_Cost(1,iter)=sum(Error.^2);
            if(Error_Cost(1,iter)<cost_gate)
                flag=1;
                break;
            else
                [   weight_input_x,...
                    weight_input_h,...
                    weight_inputgate_x,...
                    weight_inputgate_c,...
                    weight_forgetgate_x,...
                    weight_forgetgate_c,...
                    weight_outputgate_x,...
                    weight_outputgate_c,...
                    weight_preh_h ]=LSTM_updata_weight(input_num,cell_num,output_num,m,yita,Error,...
                                                       weight_input_x,...
                                                       weight_input_h,...
                                                       weight_inputgate_x,...
                                                       weight_inputgate_c,...
                                                       weight_forgetgate_x,...
                                                       weight_forgetgate_c,...
                                                       weight_outputgate_x,...
                                                       weight_outputgate_c,...
                                                       weight_preh_h,...
                                                       cell_state,h_state,...
                                                       input_gate,forget_gate,...
                                                       output_gate,gate,...
                                                       train_data,pre_h_state,...
                                                       input_gate_input,...
                                                       output_gate_input,...
                                                       forget_gate_input);

            end
        end
        if(Error_Cost(1,iter)<cost_gate)
            break;
        end
    end
    %% 绘制Error-Cost曲线图
    % for n=1:1:iter
    %     text(n,Error_Cost(1,n),'*');
    %     axis([0,iter,0,1]);
    %     title('Error-Cost曲线图');   
    % end
    %for n=1:1:iter
    %    semilogy(n,Error_Cost(1,n),'*');
    %    hold on;
    %    title('Error-Cost曲线图');   
    %end
    %% 使用第七天数据检验
    %数据加载
    %test_x
    % test_final=[0.4557 0.4790 0.7019 0.8211 0.4601 0.4811 0.7101 0.8298 0.4612 0.4845 0.7188 0.8312]';
    % test_final=test_final/sqrt(sum(test_final.^2));
    % real_test_y=test_data(:,4);
    %前馈
    m=trainNum+1;
    gate=tanh(test_final'*weight_input_x+h_state(:,m-1)'*weight_input_h);
    input_gate_input=test_final'*weight_inputgate_x+cell_state(:,m-1)'*weight_inputgate_c+bias_input_gate;
    forget_gate_input=test_final'*weight_forgetgate_x+cell_state(:,m-1)'*weight_forgetgate_c+bias_forget_gate;
    output_gate_input=test_final'*weight_outputgate_x+cell_state(:,m-1)'*weight_outputgate_c+bias_output_gate;
    for n=1:cell_num
        input_gate(1,n)=1/(1+exp(-input_gate_input(1,n)));
        forget_gate(1,n)=1/(1+exp(-forget_gate_input(1,n)));
        output_gate(1,n)=1/(1+exp(-output_gate_input(1,n)));
    end
    cell_state_test=(input_gate.*gate+cell_state(:,m-1)'.*forget_gate)';
    pre_h_state=tanh(cell_state_test').*output_gate;
    %test_y
    predict_test_y=(pre_h_state*weight_preh_h)';
    %errRate = (predict_test_y-real_test_y)/real_test_y;    %未归一化的相对误差
    %r_predict_test_y = 0.74*predict_test_y+0.22;
    %r_real_test_y = 0.74*real_test_y+0.22;
    r_predict_test_y = predict_test_y;
    r_real_test_y = real_test_y;
    r_errRate = (r_predict_test_y-r_real_test_y)/r_real_test_y;
    hit = r_predict_test_y-r_real_test_y;
    %record(:,beginer) = [predict_test_y;real_test_y;errRate;r_predict_test_y;r_real_test_y;r_errRate;hit];
    record(:,beginer) = [r_predict_test_y;r_real_test_y;r_errRate;hit];
    save('Record','record')
    clearvars -except trainNum Beginer record
end
load Record
record(5,1) = trainNum;
hitRatio = sum(record(4,:)<0.1)/size(record,2);
record(6,1) = hitRatio;
save(['Record-',num2str(Beginer),'train-',num2str(trainNum),'delay-',num2str(hitRatio),'hit.mat'],'record')