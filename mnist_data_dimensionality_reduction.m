addpath('libsvm');
p = inputParser;

data = csvread ('digits.csv');
digits = data( : , 1:400);
labels = data( : , 401:401);

%%% Picking favorite digits and creating subset
num_dig1 = 0;
num_dig2 = 0;
count_digits = 0;
for i = 1:5000;
    if labels(i, : ) == 3  ||  labels( i, : ) == 8
            count_digits = count_digits + 1;
    end
end

%%% Labels of subsets
sub_digit_labels = zeros(1,count_digits);

index = 1;
for i = 1:5000
    if labels(i, : ) == 3
        num_dig1 = num_dig1 + 1;
        sub_digit_labels(1,index) = 3;
        index = index + 1;
    elseif labels(i, : ) == 8
        num_dig2 = num_dig2 + 1;
        sub_digit_labels(1,index) = 8;
        index = index + 1;
    end
end


%%% digits of subsets
sub_digits = zeros(989, 400);

ind = 1;
for i = 1:5000
    if labels(i, : ) == 3
        sub_digits(ind,:) = digits(i,:);
        ind = ind + 1;
    elseif labels(i,: ) == 8
        sub_digits(ind,:) = digits(i,:);
        ind = ind + 1;
    end
end

sub_digits1 = zeros(500, 400);
sub_digits2 = zeros(489,400);

index = 1;
for i = 1:989
    if labels(i, : ) == 3
        sub_digits1(index,:)= digits(i,:);
        index  = index  + 1;
    elseif labels(i,: ) == 8
        sub_digits2(index,:)= digits(i,:);
        index  = index  + 1;
    end
end


%%% Computing PCA and descending order eigenvalues
sub_data = sub_digits;
[M,N] = size(sub_data);
% subtract off the mean for each dimension
mn = mean(sub_data,2);
sub_data = sub_data - repmat(mn,1,N);
% calculate the covariance matrix
covariance = (sub_data'*sub_data)./(size(sub_data,1)-1);
% find the eigenvectors and eigenvalues
[PC, V] = eig(covariance);
% extract diagonal of matrix as vector
V = diag(V);
% sort the variances in decreasing order
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
%plot(V, 'o');

%%%Display the sample mean of the dataset as an image. Also display the sample mean images for each digit

% for i = 1:5000
%     I = sub_digits(i, : );
%     mean(imagesc(reshape(I, 20, 20 )));
%     colormap(gray);
%     axis image;
% end
% % 
% for i = 1:5000 
%     if labels(i, : ) == 3
%         I = digits(i, : );
%         mean(imagesc(reshape(I, 20, 20 )));
%         colormap(gray)
%         axis image;
%     end
% end
% 
% for i = 1:5000
%     if labels(i, : ) == 8
%         I = digits(i, : );
%         mean(imagesc(reshape(I, 20, 20 )));
%         colormap(gray);
%         axis image;
%     end
% end



%%% Selecting top five eigenvectors with highest eigenvalues 
%%% and displaying top five eigenvectors images

[V1,D1]= eigs(covariance, 1, 'lm');
[V2,D2]= eigs(covariance, 2, 'lm');
[V3,D3]= eigs(covariance, 3, 'lm');
[V4,D4]= eigs(covariance, 4, 'lm');
[V5,D5]= eigs(covariance, 5, 'lm');

for i = 1
    I = V1(:, i );
    imagesc(reshape(I, 20, 20 ));
    colormap(gray);
    axis image;
end

for i = 2
    I = V2(:, i );
    imagesc(reshape(I, 20, 20 ));
    colormap(gray);
    axis image;
end

for i = 3
    I = V3(:, i );
    imagesc(reshape(I, 20, 20 ));
    colormap(gray);
    axis image;
end

for i = 4
    I = V4(:, i );
    imagesc(reshape(I, 20, 20 ));
    colormap(gray);
    axis image;
end

for i = 5
    I = V5(:, i );
    imagesc(reshape(I, 20, 20 ));
    colormap(gray);
    axis image;
end

% project the original data set
figure;
hold on;
signal1 = sub_data * V1;
signal2 = sub_data * V2;
% for i = 1:989
%     if labels(i,: ) == 3
%         plot(signal1(i,:),signal2(i,:), 'o', 'color', 'y');
%     elseif labels(i,: ) == 8
%         plot(signal1(i,:),signal2(i,:), 'x', 'color', 'k');
%     end
% end


% 
% %  Generate a histogram of the classes on projected dataset
% %  Generate another histogram of the classes on the first PCA component
% 

%histogram(projection);
histogram(V1);