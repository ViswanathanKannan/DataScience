% My first Standalone Program

clear ; close all; clc

% Load and Visualize data
fileContents = dlmread('Data_Viswa.csv',',',1,4);
fileContent = fileContents;
%(randperm(rows(fileContents)),:);

trainCount = round(rows(fileContent)* 70/100);
cvCount = round((rows(fileContent)- trainCount)/2);

X_Train = fileContent(1:trainCount,[1,2,3,4,5,6,7,8]);
y_Train = fileContent(1:trainCount,columns(fileContent));

X_CV = fileContent(trainCount+1:trainCount + cvCount,[1,2,3,4,5,6,7,8]);
y_CV = fileContent(trainCount+1:trainCount + cvCount,columns(fileContent));

X_test = fileContent(trainCount+cvCount+1:rows(fileContent),[1,2,3,4,5,6,7,8]);
y_test = fileContent(trainCount+cvCount+1:rows(fileContent),columns(fileContent));


% Find Indices of Positive and Negative Examples 
 pos = find(y_Train==1); neg = find(y_Train == 0);

normalized_X = featureNormalize(X_Train);

% Add the intercept term
normalized_Xtrain = [ones(rows(normalized_X),1) normalized_X];

% Plot Examples 
% figure;hold on;
% plot(X_Train(pos, 4), X_Train(pos, 7), 'k+', 'Color', 'b','MarkerSize', 7, 'LineWidth', 2);
% plot(X_Train(neg, 4), X_Train(neg, 7), 'ko', 'Color', 'y','MarkerSize', 7, 'LineWidth', 2);

% hold off;
%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

initial_theta = zeros(columns(normalized_Xtrain),1);

% Set regularization parameter lambda to 1
lambda = 30;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, normalized_Xtrain, y_Train, lambda)), initial_theta, options);


normalized_XCV = featureNormalize(X_CV);

% Add the intercept term
normalized_X_CV = [ones(rows(normalized_XCV),1) normalized_XCV];

p = predict(theta,normalized_X_CV);
fprintf('Train Accuracy: %f\n', mean(double(p == y_CV)) * 100);

 fp = sum((p==1) & (y_CV== 0))
 tp = sum((p==1) & (y_CV== 1))
 fn = sum((p==0) & (y_CV== 1))
 tn =sum((p==0) & (y_CV== 0))


normalized_Xtest = featureNormalize(X_test);

% Add the intercept term
normalized_X_test = [ones(rows(normalized_Xtest),1) normalized_Xtest];

p = predict(theta,normalized_X_test);
fprintf('Train Accuracy: %f\n', mean(double(p == y_test)) * 100);

 fp = sum((p==1) & (y_test== 0))
 tp = sum((p==1) & (y_test== 1))
 fn = sum((p==0) & (y_test== 1))
 tn =sum((p==0) & (y_test == 0))

 if (tp+fp)>0
   prec = (tp)/(tp+fp)
 else
   prec=0;
 end
 if (tp+fn)>0
   rec = (tp)/(tp+fn)
 else
   rec=0;
 end
 if(prec+rec) > 0
    F1 = (2 * prec*rec)/(prec+rec)
 else
    F1=0;
 end