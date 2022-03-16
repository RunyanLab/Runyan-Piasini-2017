function [new_sound_fields,new_sound_field_ids] = get_sound_features_v7(input_matrix,numfilts,post,type);
new_sound_fields = [];
new_sound_field_ids = {};
% sound_on = zeros(1,size(the_matrix,2));
% sound_off = zeros(1,size(the_matrix,2));
% a = find(diff(the_matrix(6,:))>0);
% b = find(diff(the_matrix(6,:))<0);
locations = [-90 -60 -30 -15 15 30 60 90];
% sound_on(a) = 1;
% sound_off(b) = 1;
% 
% temp = conv_sound_fields(sound_on);
% [new_sound_fields,new_sound_field_ids] = add_filtered_features(new_sound_fields,new_sound_field_ids,temp,'Sound On');
% temp = conv_sound_fields(sound_off);
% [new_sound_fields,new_sound_field_ids] = add_filtered_features(new_sound_fields,new_sound_field_ids,temp,'Sound Off');

sound_locs = zeros(8,size(input_matrix,2));
for locs = 1:8;
    %     a = find(diff(the_matrix(7+locs,:))>0);
    %     sound_locs(locs,a) = the_matrix(7+locs,a+1);
    sound_locs(locs,:) = input_matrix(locs,:);
    %     numfilts = 16;
    %     temp = conv_sound_fields(sound_locs(locs,:));
    temp = conv_any_signal_v6(sound_locs(locs,:),0,post,type,numfilts);
    [new_sound_fields,new_sound_field_ids] = add_filtered_features(new_sound_fields,new_sound_field_ids,temp,['location ' num2str(locations(locs))]);
end
%
% sound_ids = zeros(size(the_matrix,1)-6-locs,size(the_matrix,2));
% for ids = 1:size(sound_ids,1);
%     a = find(diff(the_matrix(6+locs+ids,:))>0);
%     sound_ids(ids,a) = the_matrix(6+locs+ids,a+1);
% %     temp = conv_sound_fields(sound_ids(ids,:));
%     temp = conv_any_signal(sound_ids(ids,:),0,4);
%     [new_sound_fields,new_sound_field_ids] = add_filtered_features(new_sound_fields,new_sound_field_ids,temp,['Sound ID ' num2str(ids)]);
% end
