function call_glmnet_behav_orchestra_binomial_r(base,cel)

load([base 'combined_response.mat'])
load([base 'behav_big_matrix.mat'])
load([base 'behav_big_matrix_ids.mat'])
load([base 'fold_ids.mat'])
%{
path1 = getenv('PATH')
path1 = [path1 ':/usr/local/bin/R']
setenv('PATH', path1)
%}
% threshold = 0.015;
threshold = 0;
response = combined_response>threshold;

% lsfid = str2num(getenv('LSB_JOBINDEX'));
% cel = lsfid;
features = 1:size(behav_big_matrix,1);

response_sample = response(cel,:);

opts.alpha = 0.95;
opts.lambda_min = exp(-6);
options = glmnetSet(opts);

CVerr = cvglmnetR(behav_big_matrix(features,:)',response_sample','binomial',options,'deviance',length(unique(fold_ids)),fold_ids,0)
single_glmnet.cel = cel;
single_glmnet.Beta = cvglmnetCoef(CVerr,'lambda_min');
single_glmnet.Info = CVerr;
single_glmnet.features = behav_big_matrix_ids(features);
save([base 'single_glmnet_behav_binomial_' num2str(cel)], 'single_glmnet')


% (x, y, family, options, type, nfolds, foldid, parallel)
