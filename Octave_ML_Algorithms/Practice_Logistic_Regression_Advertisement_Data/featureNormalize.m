function normalized_X = featureNormalize(X)
% Feature normalize the fields in the data and return values closer 0 to 1

mean_X = mean(X);
sigma = std(X);

normalized_X = (X - mean_X)./sigma;

