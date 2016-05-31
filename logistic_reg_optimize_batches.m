%% Optimizing classifier with valid test
% optimizing number of iterations

function optimize_batches (batch_size, initial_theta, X, y, lambda, test_X,test_y) 

lambda = 10;
initial_theta = zeros(size(X, 2), 1);
m = length(y);
alpha = 0.3;

step = 10;

batch_size = [100, 200, 300, 400];

for b = 1:length(batch_size)

    total_size = size(X,1);
    batch_num(b) = ceil(total_size / batch_size(b));
    J_history = zeros(batch_num(b), 1);

    for curr = 1:batch_num(b)
         X_curr = X(curr:curr+step,:);
         y_curr = y(curr:curr+step,:);
         test_curr_X = test_X(curr:curr+step,:);
         test_curr_y = test_y(curr:curr+step,:);

         for i = 1:m
            [J, gradJ] = lrCostFunction(initial_theta, X_curr, y_curr , lambda);
            initial_theta = initial_theta - alpha * gradJ;
         end

     J_history(curr) = J;
     fprintf('Iteration #%d - Cost = %d... \r\n',curr, J);
     
     figure;
     plot(1:batch_num(b), J_history(1:batch_num(b)), '-')
     xlabel('Number of batches')
     ylabel('Cost J')
     hold on

     % Compute accuracy on our training set
     p = predict(initial_theta , X_curr);
     fprintf('Train Accuracy: %f\n', mean(double(p == test_curr_y)) * 100);

     % Compute accuracy on our test set
     p = predict(initial_theta , test_curr_X);
     fprintf('Test Accuracy: %f\n', mean(double(p == test_curr_y)) * 100);

     
    end
end    
     
     
% for iter = 1:num_iters 
%      for i = 1:m
%         [J, gradJ] = lrCostFunction(initial_theta, X, y, lambda);
%         initial_theta = initial_theta - alpha * gradJ;
%      end
% 
%      J_history(iter) = J;
%      fprintf('Iteration #%d - Cost = %d... \r\n',iter, J);
%      
%      figure;
%      plot(1:num_iters, J_history(1:num_iters), '-')
%      xlabel('Number of iterations')
%      ylabel('Cost J')
%      hold on
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



