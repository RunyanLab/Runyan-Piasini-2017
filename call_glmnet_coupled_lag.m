function call_glmnet_combinedwpopmean_orchestra_binomial_lag_r(base,filters_to_keep,cel)

load([base 'combined_response.mat'])
% load([base 'cells_big_matrix.mat'])
load([base 'behav_big_matrix.mat'])
% load([base 'cells_big_matrix_ids.mat'])
load([base 'behav_big_matrix_ids.mat'])
load([base 'fold_ids.mat'])

% lsfid = str2num(getenv('LSB_JOBINDEX'));
% cel = lsfid;

threshold = 0;
response = combined_response>threshold;

[cells_big_matrix,cells_big_matrix_ids] = make_big_matrix_lags(combined_response,filters_to_keep);
big_matrix_temp = cell2mat(cells_big_matrix_ids);
cell_features = find(big_matrix_temp~=cel);
behav_inds = 1:size(behav_big_matrix,1);
cell_inds = size(behav_big_matrix,1)+1:size(behav_big_matrix,1)+length(cell_features);
big_matrix = cat(1,behav_big_matrix,cells_big_matrix(cell_features,:));
big_matrix_ids = [behav_big_matrix_ids cells_big_matrix_ids(cell_features)];
popind1 = size(big_matrix,1)+1;
temp = 1:size(response,1);
inds = find(temp~=cel);
popmean = mean(response(inds,:));

[big_matrixpop,big_matrix_idspop] = make_big_matrix_lags(popmean,filters_to_keep);
big_matrix = cat(1,big_matrix,big_matrixpop);
big_matrix_ids(end+1:size(big_matrix,1)) = repmat({'POP Mean'},size(big_matrix,1)-length(big_matrix_ids),1);

features = 1:size(big_matrix,1);

response_sample = response(cel,:);

opts.alpha = 0.95;
shrinkage = ones(length(features),1);
shrinkage(cell_inds) = 10;
shrinkage(popind1:end) = 20;
opts.penalty_factor = shrinkage;
options = glmnetSet(opts);

CVerr = cvglmnetR(big_matrix(features,:)',response_sample','binomial',options,'deviance',length(unique(fold_ids)),fold_ids,0)

single_glmnet.cel = cel;
single_glmnet.Beta = cvglmnetCoef(CVerr,'lambda_min');
single_glmnet.Info = CVerr;
single_glmnet.features = big_matrix_ids(features);
single_glmnet.shrinkage = shrinkage;

save([base 'single_glmnet_combinedwpopmean_binomial_' num2str(cel), 'single_glmnet lag ' num2str(filters_to_keep-20)],'single_glmnet')
