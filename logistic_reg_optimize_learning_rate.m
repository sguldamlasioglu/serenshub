%% Optimizing classifier with valid test
% optimizing learning_rate

function optimize_learning_rate (initial_theta, X, y,lambda, test_X, test_y)

alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3];  % learning rate
J_history = zeros(length(alpha), 1);
m = length(y);
lambda = 2;

for a = 1:length(alpha)
 
    for i = 1:m
            [J, grad] = lrCostFunction(initial_theta, X, y, lambda);
            initial_theta = initial_theta - alpha(a) * grad;      
    end
  
    J_history(a) = J;
    fprintf('Alpha #%d - Cost = %d... \r\n',alpha(a), J);
    
    %Compute accuracy on our training set
      p = predict(initial_theta , X);
      fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

    %Compute accuracy on our test set
     p = predict(initial_theta , test_X);
     fprintf('Test Accuracy: %f\n', mean(double(p == test_y)) * 100);

    figure;
    plot(alpha, J_history(1:length(alpha)), '-')
    xlabel('Alpha')
    ylabel('Cost J')
    hold on
   
end        
end







