%% Optimizing classifier with valid test
% optimizing regularization

function optimize_regularization (initial_theta, X, y,lambda, test_X, test_y)

lambda = [1, 10, 100, 200];  % learning rate
J_history = zeros(length(lambda), 1);
m = length(y);
alpha = 0.3;

for l = 1:length(lambda)
 
    for i = 1:m
            [J, grad] = lrCostFunction(initial_theta, X, y, lambda(l));
            initial_theta = initial_theta - alpha * grad;
          
    end
  
    
    J_history(l) = J;
    fprintf('Lambda #%d - Cost = %d... \r\n',lambda(l), J);
    
    %Compute accuracy on our training set
      p = predict(initial_theta , X);
      fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

    %Compute accuracy on our test set
     p = predict(initial_theta , test_X);
     fprintf('Test Accuracy: %f\n', mean(double(p == test_y)) * 100);

    figure;
    plot(lambda, J_history(1:length(lambda)), '-')
    xlabel('Lambda')
    ylabel('Cost J')
    hold on
   
end        
end







