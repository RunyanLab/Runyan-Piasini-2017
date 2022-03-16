function [big_matrix,big_matrix_id] = add_filtered_features(big_matrix,big_matrix_id,new_features,new_id);

pre_inds = size(big_matrix,1);
new_inds = size(new_features,1);

big_matrix(pre_inds+1:pre_inds+new_inds,:) = new_features;

if iscell(new_id)==1
    big_matrix_id(pre_inds+1:pre_inds+new_inds) = new_id;
else
    if ischar(new_id)==1
    big_matrix_id(pre_inds+1:pre_inds+new_inds) = {new_id};
    else
         big_matrix_id(pre_inds+1:pre_inds+new_inds) = mat2cell(new_id,1,1);
    end
    
end
