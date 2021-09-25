result1 = [];
result2 = [];
resultcv = [];
for i= 1:14
   %training data;
    load Actual_training.txt;
    train=Actual_training;
    clear Actual_training;
    label_train = train(:,1);
    
gscatter(train(:,3),train(:,4),label_train);
 b=legend('Soyabean-notill','Woods','Soybean-min','Corn-min','Corn-notill');
set(b,'Location','SouthEast');
xlabel('Feature 1');
ylabel('Feature 2');

    train(:,1:2)=[];
    train = train(:,1:i);
    mx_train = max(train(:));
    mn_train = min(train(:));
    
    %test data;
    load Actual_test.txt;
    test = Actual_test;
    clear Actual_test;
    label_test = test(:,1);
    test(:,1:2)=[];
    test = test(:,1:i);
    mx_test = (max(test(:)));
    mn_test = min(test(:));
    
    train = (train-mn_train) ./ (mx_train-mn_train);
    test = (test-mn_test) ./ (mx_test-mn_test);

    addpath('I:\libsvm-3.22\matlab');
    
  bestc=4; bestg=0  ; 
 
    
%   bestcv=0; bestc=0; bestg=0;
%    for c = 0:1:5
%       for g = 0:0.5:8
%             cmd=['-t 2 -v 10 -c ',num2str(c), ' -g 4', num2str(g)];
%            cv = svmtrain(label_train, train, cmd);
%          if(cv>bestcv)
%                bestcv=cv; bestc=c; bestg=g;
%             end
%             fprintf('%g   %g  %g (best c=%g, g=%g, rate=%g)\n', c, g, cv, bestc, bestg, bestcv);
%        end
%      end
%      
%     resultcv = [resultcv bestcv];
    cmd=['-t 2 -c ',num2str(bestc), ' -g 4', num2str(bestg)];
    
    
   model = svmtrain(label_train,train,cmd);
%     fprintf('\n\nTraining Data Accuracy: \n\n ');
%     [predict_label, trainingaccuracy, dec_values]= svmpredict(label_train, train, model);
%     result1 = [result1 trainingaccuracy(1)];
   
    fprintf('\n\nTest Data Accuracy: \n\n ');
    [predict_label, testaccuracy, dec_values] = svmpredict(label_test, test, model);
    result2 = [result2 testaccuracy(1)];
end


