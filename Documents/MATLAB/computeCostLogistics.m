function [ J ] = computeCostLogistics( X, y, theta )


% Prepare Variables
    m = length(y);
    
    % Calculate Hypothesis
    h = X * theta;
    
    % Calculate Cost
    J = (-1 / m) * (sum(y .* log(h) + (ones(m,1) - y) .* log(h)));

end

