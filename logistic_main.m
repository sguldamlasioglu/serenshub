%% Initialization
clear ; close all; clc

%% Load Data
data = csvread ('digits.csv');
digits = data( : , 1:400);
labels = data( : , 401:401);

%% Converting to Even and Odd Digits
for item = 1:length(labels);
    if mod(labels(item,1),2) == 0;   % Even
        labels(item,1) = 0;
    else mod(labels(item,1),2) == 1;  % Odd
        labels(item,1) = 1;
    end
end
 
%% Creating Training and Test Set 
[trainInd,validInd] = dividerand(5000,0.7,0.3);

X = digits(1:3044, :); 
test_X = digits(3044:5000, :);

y = labels(1:3044,:);
test_y = labels(3044:5000,:);   


%% Optimize Number of Iteration

% initial_theta = zeros(size(X, 2), 1);
% lambda = 10;
% 
% optimize_iteration (initial_theta, X, y,lambda, test_X, test_y) 


%% Optimize Learning Rate

% alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3]; 
% initial_theta = zeros(size(X, 2), 1);
% lambda = 10;
% m = length(y);
% 
% optimize_learning_rate (initial_theta, X, y,lambda, test_X, test_y);



%% Optimize Regularization Weight

% lambda = [1, 10, 100, 200];
% initial_theta = zeros(size(X, 2), 1);
% m = length(y);
% alpha = 0.3;
% 
% optimize_regularization(initial_theta, X, y,lambda, test_X, test_y);



%% Optimize Batch Size

% lambda = 10;
% initial_theta = zeros(size(X, 2), 1);
% m = length(y);
% alpha = 0.3;
% batch_size = [100, 200, 300, 400];
% 
% optimize_batches (batch_size, initial_theta, X, y, lambda, test_X,test_y)


%% Part 3/A - Final Set of Parameters

% lambda = 10;
% alpha = 0.3;
% batch_size = 100;
% num_iters= 20;
% m = length(y);
% initial_theta = zeros(size(X, 2), 1);
% step = 10;
% 
% total_size = size(X,1);
% batch_num = ceil(total_size / batch_size);
% J_history = zeros(num_iters, 1);
% 
% for iter = 1:num_iters 
%     for curr = 1:batch_num
%         
%          X_curr = X(curr:curr+step,:);
%          y_curr = y(curr:curr+step,:);
%          test_curr_X = test_X(curr:curr+step,:);
%          test_curr_y = test_y(curr:curr+step,:);
% 
%          for i = 1:m
%             [J, gradJ] = lrCostFunction(initial_theta, X_curr, y_curr , lambda);
%             initial_theta = initial_theta - alpha * gradJ;
%          end
%     end
%      J_history(iter) = J;
%      %  fprintf('Iteration #%d - Cost = %d... \r\n',iter, J);
%      
% %      figure;
% %      plot(1:num_iters, J_history(1:num_iters), '-')
% %      xlabel('Number of iterations')
% %      ylabel('Cost J')
% %      hold on
% 
%      % Compute accuracy on our training set
%      p = predict(initial_theta , X);
%      fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
% 
%      % Compute accuracy on our test set
%      p = predict(initial_theta , test_X);
%      fprintf('Test Accuracy: %f\n', mean(double(p == test_y)) * 100);
%     
% 
% end


 
%% Plot for number of iterations

% initial_theta = zeros(size(X, 2), 1);
% num_iters= 25;  % num_iters
% alpha = 0.3;  % constant learning rate
% m = length(y);
% lambda = 10;
% batch_size = 100;
% test_acc = zeros(25,1);
% train_acc = zeros(25,1);
% 


% total_size = size(X,1);
% batch_num = ceil(total_size / batch_size);
% J_history = zeros(batch_num, 1);
% step = 10;


% for iter = 1:num_iters
%     
%     for curr = 1:batch_num
%     
%      X_curr = X(curr:curr+step,:);
%      y_curr = y(curr:curr+step,:);
%      test_curr_X = test_X(curr:curr+step,:);
%      test_curr_y = test_y(curr:curr+step,:);
%     
%         for i = 1:m
%             [J, grad] = lrCostFunction(initial_theta, X_curr, y_curr, alpha);
%             initial_theta = initial_theta - alpha * grad;
%         end
%          
%     end
%      J_history(iter) = J;
%      %fprintf('A #%d - Cost = %d... \r\n',iter, J)
% 
%      % Compute accuracy on our training set
%      
%      p = predict(initial_theta , X_curr);
%      train_acc(iter) =  mean(double(p == y_curr)) * 100;
%      %fprintf('Train Accuracy: %f\n', train_acc);
% 
%      % Compute accuracy on our test set
%      
%      p = predict(initial_theta , test_curr_X);
%      test_acc(iter) = mean(double(p == test_curr_y)) * 100;
%      %fprintf('Test Accuracy: %f\n', test_acc);
%      
%  end  
% 
% 
%  figure;
%  plot(1:num_iters,train_acc,'b',1:num_iters,test_acc,'r');
%  xlabel('Number of Iteration')
%  ylabel('Accuracy')
%  hold on


%% Plot for batch sizes

% initial_theta = zeros(size(X, 2), 1);
% alpha = 0.3;  % constant learning rate
% 
% m = length(y);
% lambda = 10;
% batch_size = [100, 200, 300, 400];
% 
% test_acc = zeros(4,1);
% train_acc = zeros(4,1);
% 
% step = 10;
% 
% for b = 1:length(batch_size)
% 
%     total_size = size(X,1);
%     batch_num(b) = ceil(total_size / batch_size(b));
%     
%     J_history = zeros(batch_num(b), 1);
%     
%     for curr = 1:batch_num(b)
%     
%      X_curr = X(curr:curr+step,:);
%      y_curr = y(curr:curr+step,:);
%      test_curr_X = test_X(curr:curr+step,:);
%      test_curr_y = test_y(curr:curr+step,:);
%     
%         for i = 1:m
%             [J, grad] = lrCostFunction(initial_theta, X_curr, y_curr, alpha);
%             initial_theta = initial_theta - alpha * grad;
%         end
%          
%     end
%      J_history(b) = J;
%      %fprintf('A #%d - Cost = %d... \r\n',iter, J)
% 
%      % Compute accuracy on our training set
%      
%      p = predict(initial_theta , X_curr);
%      train_acc(curr) =  mean(double(p == y_curr)) * 100;
%      %fprintf('Train Accuracy: %f\n', train_acc);
% 
%      % Compute accuracy on our test set
%      
%      p = predict(initial_theta , test_curr_X);
%      test_acc(curr) = mean(double(p == test_curr_y)) * 100;
%      %fprintf('Test Accuracy: %f\n', test_acc);
%      
%  end  
% 
% 
%  figure;
%  plot(1:batch_num,train_acc,'b',1:batch_num,test_acc,'r');
%  xlabel('Batch Size')
%  ylabel('Accuracy')
%  hold on



%% Plot for for regularization weight 

% initial_theta = zeros(size(X, 2), 1);
% num_iters= 1;  % num_iters
% lambda = [1, 10, 30, 50, 80, 100, 200];
% alpha = 0.03;  % constant learning rate
% batch = 100;
% J_history = zeros(length(lambda), 1);
% m = length(y);
% test_acc = zeros(7,1);
% train_acc = zeros(7,1);
% 
% for l = 1:length(lambda)
%     for i = 1:m
%         [J, grad] = lrCostFunction(initial_theta, X, y, lambda(l));
%         initial_theta = initial_theta - alpha * grad;
%     end
%          
%      J_history(l) = J;
%      %fprintf('Lambda #%d - Cost = %d... \r\n',lambda(l), J)
% 
%      % Compute accuracy on our training set
%      
%      p = predict(initial_theta , X);
%      train_acc(l) =  mean(double(p == y)) * 100;
%      %fprintf('Train Accuracy: %f\n', train_acc);
% 
%      % Compute accuracy on our test set
%      
%      p = predict(initial_theta , test_X);
%      test_acc(l) = mean(double(p == test_y)) * 100;
%      %fprintf('Test Accuracy: %f\n', test_acc);
%      
% end  
% 
%  figure;
%  plot(lambda,train_acc,'b',lambda,test_acc,'r');
%  xlabel('Lambda')
%  ylabel('Accuracy')
%  hold on






%% Plot for learning rate

% initial_theta = zeros(size(X, 2), 1);
% num_iters= 1;  % num_iters
% alpha = [0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.3];  % constant learning rate
% J_history = zeros(length(alpha), 1);
% m = length(y);
% lambda = 10;
% batch = 100;
% test_acc = zeros(7,1);
% train_acc = zeros(7,1);
% 
% for a = 1:length(alpha)
%     for i = 1:m
%         [J, grad] = lrCostFunction(initial_theta, X, y, alpha(a));
%         initial_theta = initial_theta - alpha(a) * grad;
%     end
%          
%      J_history(a) = J;
%      %fprintf('A #%d - Cost = %d... \r\n',alpha(a), J)
% 
%      % Compute accuracy on our training set
%      
%      p = predict(initial_theta , X);
%      train_acc(a) =  mean(double(p == y)) * 100;
%      %fprintf('Train Accuracy: %f\n', train_acc);
% 
%      % Compute accuracy on our test set
%      
%      p = predict(initial_theta , test_X);
%      test_acc(a) = mean(double(p == test_y)) * 100;
%      %fprintf('Test Accuracy: %f\n', test_acc);
%      
% end  
% 
%  figure;
%  plot(alpha,train_acc,'b',alpha,test_acc,'r');
%  xlabel('Learning Rate')
%  ylabel('Accuracy')
%  hold on


