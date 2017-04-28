clear all;
clc
close all;

prompt='Enter the CandidateID to find his/her future - ';
user_number_skill=input(prompt);


tic

 Y = [1 0 0 0 0 1 0 1 1 1;0 1 0 0 0 1 1 1 0 1;0 0 1 0 0 0 1 1 1 0;
     0 0 0 1 0 1 0 1 1 0;0 0 0 0 1 1 0 1 1 1;0 1 0 0 0 1 1 1 1 1;
     1 0 0 0 0 1 0 1 1 1;0 0 0 0 0 1 1 1 1 0;0 0 0 0 0 0 0 1 1 1;
     0 0 0 0 0 1 0 1 1 0];
%Y=randi([0,10],10,10);

R=Y;

%  Y is a 10X10 matrix, containing ratings (0 or 1) of 10 skills on 
%  10 users
%
%  R is a 10X10 matrix, where R(i,j) = 1 if and only if user j have
%  that skill i
%  We can "visualize" the ratings matrix by plotting it with imagesc

imagesc(Y);
ylabel('Skill');
xlabel('Users');

%enter the candidate id to find his/her future skill

y_u_s=zeros(10,1);
y_u_s=Y(:,user_number_skill);

%  Collaborative Filtering Cost Function 
%  we will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in 
%  cofiCostFunc.m to return J
%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
%  X is experiance of user
%  Theta is minimum experiance that is needed

X = [2;3;2;3.6;4;3.7;3;2;2.7;2.9]
X_max=max(X)+0.2;
X = X./X_max;
Theta = [2;3;3;2;4;2;3;2;4;3]
Theta_max= max(Theta)+0.2;
Theta=Theta./Theta_max;
num_users = 10; num_skills = 10; num_features = 1;    


%  Collaborative Filtering Cost Regularization 
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  Evaluate cost function

J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_skills, ...
               num_features, 1.5);


%  Collaborative Filtering Gradient Regularization 
%  Once your cost matches up with ours, you should proceed to implement 
%  regularization for the gradient. 
%
%  Check gradients by running checkNNGradients
checkCostFunction(1.5);

%  Normalize skills
[Ynorm, Ymean] = normalizeRatings(Y, R);

% Set Initial Parameters (Theta, X)
initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 0.1;
theta = fmincg (@(t)(cofiCostFunc(t, Y, R, num_users, num_skills, num_features, lambda)), initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_skills*num_features), num_skills, num_features);
Theta = reshape(theta(num_skills*num_features+1:end), ...
                num_users, num_features);
            
            

fprintf('Recommender system learning completed.\n');


%  Recommendation for you 
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.

p = X * Theta';

%figure;
%scatter(X,p(:,1:10))
figure;
plot(X,p(:,user_number_skill))
hold on;
plot(X,Y,'o')
hold off;

%my_predictions = p(:,user_number_skill) + Ymean;

skillList = loadSkillsList();

[r, ix] = sort( p(:,user_number_skill), 'descend');


fprintf('\nTop skill recommendations for user ID %s :\n',user_number_skill);
for i = 1:10
    if y_u_s(i)>0 
        rsh=0;
    else
        j = ix(i);
        fprintf('Predicting top new skill for future : %s\n', skillList{j});
    end
end
toc