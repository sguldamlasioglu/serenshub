
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


%%%%%%% QUESTION 2 - a

zip = [sub_digits, sub_digit_labels'];

shuffle_data=zip(randsample(1:length(zip),length(zip)),:);
shuffled_digits = shuffle_data( : , 1:400);
shuffled_labels = shuffle_data( : , 401:401);

[trainInd,testInd] = dividerand(count_digits,0.7,0.3);

trainfeature = shuffled_digits(trainInd, :);
testfeature = shuffled_digits(testInd, :);

trainlabel = shuffled_labels(1:602,1);
testlabel = shuffled_labels(1:258,1);

numSamples = size(trainfeature, 1);
numFolds = 5;

cvIdx = crossvalind('Kfold', numSamples, numFolds);


for fold = 1:numFolds
    cv_train = trainfeature(cvIdx ~= fold , :);
    cv_train_label = trainlabel(cvIdx ~= fold,1);
    cv_valid = trainfeature(cvIdx == fold , :);
    cv_valid_label = trainlabel(cvIdx == fold,1);
    
    options.MaxIter = 100000; 
    acc=zeros(1,8);
    runCounter = 1;
    bestaccuracy = 0;
    index = 1;
    C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 10, 100, 1000];
    
    for s=1:8
        svm_model = svmtrain(trainfeature, trainlabel, 'boxconstraint',C(1,s),'tolkkt', 0.000001, 'kktviolationlevel', 0.7, 'Kernel_Function', 'linear', 'Options', options);
        testoutput= svmclassify(svm_model,cv_valid,'Showplot','false'); 
        accuracy = sum(cv_valid_label == testoutput) ./ numel(cv_valid_label);
        accuracy
        
        acc(1,s) = accuracy;
        
        if (accuracy >= bestaccuracy)
            bestaccuracy = accuracy;
            index = index + 1;
        end
        runCounter = runCounter+1;      
    end
     
end
 
plot(acc);

updated_model = svmtrain(trainfeature, trainlabel, 'boxconstraint',10,'tolkkt', 0.0001, 'Kernel_Function', 'linear', 'kktviolationlevel', 0.6,'Options', options);
updated_testoutput= svmclassify(updated_model,testfeature,'Showplot','false'); 
test_accuracy = sum(testlabel == updated_testoutput) ./ numel(testlabel);

fileID = fopen('linear_kernel.txt','w');
fprintf(fileID,'%6s %12s\n','predicted test label','test label');
fprintf(fileID,'%6.2f %12.8f\n',updated_testoutput, testlabel);
fclose(fileID);










 


 

