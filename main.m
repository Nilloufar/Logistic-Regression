format shortG
clc
%% load data

data =load('data.txt');
labels=load('labels.txt');

%% prepare train and test data
x_train=data(1:2000,:);
x_train=[x_train ones(2000,1)];
y_train=labels(1:2000,:);

x_test=data(2001:end,:);
x_test=[x_test ones(size(x_test,1),1)];
y_test=labels(2001:end,:);
N_test=size(y_test,1);

%% Train logistic regression classifier on the first n rows of the training
n=[200, 500, 800, 1000, 1500, 2000];

for i=1:size(n,2)
    num_train_rows= n(i);
    weights=logistic_train(x_train(1:num_train_rows,:),y_train(1:num_train_rows,:),10^-5,6000,0.1);
    predict_label= round(predict(x_test,weights));
    accuracy=size(find(predict_label==y_test),1)/N_test;
%     fprintf('training Accuracy Training with first %d rows is: %d \n',num_train_rows,accuracy*100)
    fprintf(' %d \n',accuracy*100)

end