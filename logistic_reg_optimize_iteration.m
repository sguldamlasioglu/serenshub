%% Optimizing classifier with valid test
% optimizing number of iterations

function optimize_iteration (initial_theta, X, y, lambda, test_X,test_y) 

num_iters= 40;  % num_iters
alpha = 0.3;  % constant learning rate
J_history = zeros(num_iters, 1);
m = length(y);
lambda = 10;

for iter = 1:num_iters 
     for i = 1:m
        [J, gradJ] = lrCostFunction(initial_theta, X, y, lambda);
        initial_theta = initial_theta - alpha * gradJ;
     end

     J_history(iter) = J;
     fprintf('Iteration #%d - Cost = %d... \r\n',iter, J);
     
     figure;
     plot(1:num_iters, J_history(1:num_iters), '-')
     xlabel('Number of iterations')
     ylabel('Cost J')
     hold on

     % Compute accuracy on our training set
     p = predict(initial_theta , X);
     fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

     % Compute accuracy on our test set
     p = predict(initial_theta , test_X);
     fprintf('Test Accuracy: %f\n', mean(double(p == test_y)) * 100);
    

end



