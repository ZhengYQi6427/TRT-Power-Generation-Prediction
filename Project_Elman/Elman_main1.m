clear
clc
trainNum = 3;       %ѵ��������
Beginer = 100;
load('Elman:7train-3delay-0.85714hit.mat')
record1(1:4,1:7)=record(1:4,1:7);
for beginer=1:Beginer
    %% ���ݼ���
    if beginer<=7
    load Data
    for i=1:3
        train_data(:,i) = reshape(Data((beginer+i):(trainNum+beginer+i-1),1:7),trainNum*7,1);
        test_data(:,i) = Data(trainNum+beginer+i,:)';
    end
    test_final = reshape(Data(beginer+trainNum+1:beginer+trainNum+3,1:7),trainNum*7,1);
    %[train_data,test_data]=Elman_data_process(beginer,trainNum);
    data_length=size(train_data,1);
    data_num=size(train_data,2);
    else
    for i=1:3
        train_data(:,i)=reshape(r_predict_test_y(1:7,(beginer+i-7):(trainNum+beginer+i-8)),trainNum*7,1);
        test_data(:,i)=r_predict_test_y(:,trainNum+beginer+i-7);
    end
    test_final = reshape(r_predict_test_y(1:7,beginer+trainNum+1-7:beginer+trainNum+3-7),trainNum*7,1);
    data_length=size(train_data,1);
    data_num=size(train_data,2);
    end           
    %% ���������ʼ��
    %�����Ŀ����
    input_layer_num=data_length;
    hidden_layer_num=3; %������
    output_layer_num=size(test_data,1);
    %Ȩ�س�ʼ��
    weight_input_hidden=rand(hidden_layer_num,input_layer_num)/1000;
    weight_prehidden_hidden=rand(hidden_layer_num,hidden_layer_num)/1000;
    weight_hidden_output=rand(output_layer_num,hidden_layer_num)/1000;
    output_state=zeros(output_layer_num,3);
    %% ����ѵ��ѧϰ����  ����BPTT�㷨
    yita=0.001;              %ÿ�ε����Ĳ���
    for iter=1:4000
        for t=1:3      %%ʱ�䲽
            %�����뵽����
            if (t==1)
                pre_hidden_state=weight_input_hidden*train_data(:,t);
            else
                pre_hidden_state=weight_input_hidden*train_data(:,t)+weight_prehidden_hidden*hidden_state(:,t-1);
            end
            %�����㵽���
            for n=1:hidden_layer_num
                hidden_state(n,t)=1/(1+exp(-pre_hidden_state(n,:)));              %%ͨ��sigmoid����
            end
            output_state(:,t)=weight_hidden_output*hidden_state(:,t);
            %������
            Error=output_state(:,t)-test_data(:,t);
            Error_cost(1,iter)=sum((output_state(:,t)-test_data(:,t)).^2);
            if(Error_cost(1,iter)<1e-4)
                break;
            end
            %Ȩֵ����
            [weight_input_hidden,weight_prehidden_hidden,weight_hidden_output]=updata_weight(t,yita,Error,train_data,hidden_state,weight_input_hidden,weight_prehidden_hidden,weight_hidden_output);
        end
        if(Error_cost(1,iter)<1e-4)
            break;
        end
    end
    %% ����COST����
    %for n=1:1:iter
    %   text(n,Error_cost(1,n),'*');
    %    axis([0,iter,0,1]);
    %    title('Error-cost����ͼ');
    %end
    %% �������
    %��3��5��ʱ����Ƶ�6��
    train_data(:,3);
    pre_hidden_state=weight_input_hidden*train_data(:,3)+weight_prehidden_hidden*hidden_state(:,2);
    for n=1:hidden_layer_num
        hidden_state(n,3)=1/(1+exp(-pre_hidden_state(n,:)));              %%ͨ��sigmoid����
    end
    %output_state(:,3)=weight_hidden_output*hidden_state(:,3);
    %Ԥ���7��ʱ���
    pre_hidden_state=weight_input_hidden*test_final+weight_prehidden_hidden*hidden_state(:,3);
    for n=1:hidden_layer_num
        hidden_state1(n,1)=1/(1+exp(-pre_hidden_state(n,:)));              %%ͨ��sigmoid����
    end
    t_output=weight_hidden_output*hidden_state1(:,1);
    
    r_predict_test_y(:,beginer) = t_output;
    load Data
    r_real_test_y = Data(trainNum+3+beginer,7);
    r_errRate = (r_predict_test_y-r_real_test_y)/r_real_test_y;
    hit = r_predict_test_y-r_real_test_y;
    %record(:,beginer) = [predict_test_y;real_test_y;errRate;r_predict_test_y;r_real_test_y;r_errRate;hit];
    record1(:,beginer+7) = [r_predict_test_y(7,beginer);r_real_test_y;r_errRate(7,beginer);hit(7,beginer)];   
    save('Record','record1')
    clearvars -except trainNum Beginer record1 r_predict_test_y
end 
load Record
mse = sum(record1(4,:).^2)/length(record1(1,:));
hitRatio = sum(abs(record1(4,:))<=0.1)/size(record1,2);
save(['longterm:',num2str(Beginer),'train-',num2str(trainNum),'delay-',num2str(hitRatio),'hit.mat'],'record1')