function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_skills, ...
                                  num_features, lambda)

X = reshape(params(1:num_skills*num_features), num_skills, num_features);
Theta = reshape(params(num_skills*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% only accumulate cost j if Rij = 1
Ymasked = R .* Y;


% SLOW ITERATIVE SOLUTION
tot = 0.0;
for i=1:num_skills
    for j=1:num_users
        if R(i, j) == 0
            continue
        end

        Tj = Theta(j, :);
        xi = X(i, :);
        yij = Y(i, j);

        incr = (xi * Tj' - yij) ^ 2;
        % keyboard();
        tot = tot + incr;
        % tot
    end
end

J = tot / 2;


grad = [X_grad(:); Theta_grad(:)];

end
