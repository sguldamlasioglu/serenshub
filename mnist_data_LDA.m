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

%  Use LDA to get a new base
%  Project your dataset on the base you obtained from LDA

[projection, trainedlda] = lda(sub_digits,sub_digit_labels);

% Displaying digits on LDA

% for i = 400
%     I = trainedlda.M(i,1);
%     imagesc(reshape(I, 20, 20 ));
%     colormap(gray);
%     axis image;
% end

%  histogram projection

%histogram(projection);






