function [big_matrix,big_matrix_ids,the_matrix_used,the_matrix_ids_used,response,this_trial_start,thisloc,thisfreq,thiscorrect] = make_big_matrix_final(imaging,type,filters_to_keep);
warning('off','all')


%%first, pull responses and matched "features" corresponding to the imaging
%%frames - this step is not necessary for the cell x cell model, because
%%just the response matrix will be necessary
[response,the_matrix,the_matrix_ids,this_trial_start,thisloc,thisfreq,thiscorrect] = get_the_matrix_final(imaging);

if type == 'behav'
%%next, convolve each feature with the appropriate set of basis functions,
%%and add to the big matrix.
peak_locs = [];
%%position in maze
[y_fields,y_field_ids] = make_place_fields_final(the_matrix);
big_matrix = [];
big_matrix_ids = {};
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,y_fields(end,:),y_field_ids(end));


%%now add the upcoming turns, convolved
upcoming_left = the_matrix(5,:);
upcoming_right = the_matrix(6,:);
[upcoming_left_space,upcoming_right_space] = combine_space_turn(upcoming_left,upcoming_right,y_fields(1:end-1,:));
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,upcoming_left_space,'upcoming left turn');
peak_locs(1:length(big_matrix_ids)) = 1;
numfilts = 9;
[upcoming_left_conv,temp_peak_locs] = conv_any_signal_final(upcoming_left,0,4,1,numfilts);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,upcoming_left_conv,'upcoming left turn');
peak_locs = cat(2,peak_locs,temp_peak_locs);

[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,upcoming_right_space,'upcoming right turn');
temp_peak_locs(1:size(upcoming_right_space,1)) = 1;
peak_locs = cat(2,peak_locs,temp_peak_locs);
numfilts = 9;
[upcoming_right_conv,temp_peak_locs] = conv_any_signal_final(upcoming_right,0,4,1,numfilts);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,upcoming_right_conv,'upcoming right turn');
peak_locs = cat(2,peak_locs,temp_peak_locs);

y_velocity = the_matrix(2,:);
temp = zeros(size(y_velocity));
a = find(y_velocity>0);
b = find(y_velocity<0);
neg_y_velocity = temp;
neg_y_velocity(b) = abs(y_velocity(b));
pos_y_velocity = temp;
pos_y_velocity(a) = y_velocity(a);

x_velocity = the_matrix(3,:);
temp = zeros(size(x_velocity));
a = find(x_velocity>0);
b = find(x_velocity<0);
neg_x_velocity = temp;
neg_x_velocity(b) = abs(x_velocity(b));
pos_x_velocity = temp;
pos_x_velocity(a) = x_velocity(a);

[y_velocity_conv,temp_peak_locs] = conv_any_signal_final(pos_y_velocity,-1,0,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,y_velocity_conv,'y velocity');
[y_velocity_conv,temp_peak_locs] = conv_any_signal_final(pos_y_velocity,0,1,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,y_velocity_conv,'y velocity');

[y_velocity_conv,temp_peak_locs] = conv_any_signal_final(neg_y_velocity,-1,0,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,y_velocity_conv,'y velocity');
[y_velocity_conv,temp_peak_locs] = conv_any_signal_final(neg_y_velocity,0,1,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,y_velocity_conv,'y velocity');

[x_velocity_conv,temp_peak_locs] = conv_any_signal_final(pos_x_velocity,-1,0,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,x_velocity_conv,'x velocity');
[x_velocity_conv,temp_peak_locs] = conv_any_signal_final(pos_x_velocity,0,1,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,x_velocity_conv,'x velocity');

[x_velocity_conv,temp_peak_locs] = conv_any_signal_final(neg_x_velocity,-1,0,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,x_velocity_conv,'x velocity');
[x_velocity_conv,temp_peak_locs] = conv_any_signal_final(neg_x_velocity,0,1,1,4);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,x_velocity_conv,'x velocity');


view_angle = the_matrix(4,:);
view_angle = rad2deg(view_angle)-90;
temp = zeros(size(view_angle));
a = find(view_angle>0);
b = find(view_angle<0);
neg_view_angle = temp;
neg_view_angle(b) = abs(view_angle(b));
pos_view_angle = temp;
pos_view_angle(a) = view_angle(a);

[view_angle_conv,temp_peak_locs] = conv_any_signal_final(pos_view_angle,-1,0,1,3);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,view_angle_conv,'view angle');
[view_angle_conv,temp_peak_locs] = conv_any_signal_final(pos_view_angle,0,1,1,3);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,view_angle_conv,'view angle');

[view_angle_conv,temp_peak_locs] = conv_any_signal_final(neg_view_angle,-1,0,1,3);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,view_angle_conv,'view angle');
[view_angle_conv,temp_peak_locs] = conv_any_signal_final(neg_view_angle,0,1,1,3);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,view_angle_conv,'view angle');

%next add reward, conv only to min ITI length, and 0.5 sec before

reward_correct = the_matrix(7,:);
reward_incorrect = the_matrix(8,:);
% 
% numfilts = 2;
% reward_conv = conv_any_signal_final(reward_correct,-1,0,1,numfilts);
% [big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,reward_conv,'reward');
numfilts = 4;
[reward_conv,temp_peak_locs] = conv_any_signal_final(reward_correct,0,2,1,numfilts);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,reward_conv,'reward');

% numfilts = 2;
% reward_conv = conv_any_signal_final(reward_incorrect,-1,0,1,numfilts);
% [big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,reward_conv,'noreward');
numfilts = 4;
[reward_conv,temp_peak_locs] = conv_any_signal_final(reward_incorrect,0,2,1,numfilts);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,reward_conv,'noreward');

%%add the sound features
[sound_fields,sound_field_ids,temp_peak_locs] = get_sound_features_v7(the_matrix(9:16,:),12,2.1,1);
peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,sound_fields,sound_field_ids);

[sound_fields,sound_field_ids,temp_peak_locs] = get_sound_features_v7(the_matrix(17:24,:),12,2.1,1);
peak_locs = cat(2,peak_locs,temp_peak_locs);

[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,sound_fields,sound_field_ids);

[sound_fields,sound_field_ids,temp_peak_locs] = get_sound_features_v7(the_matrix(25:32,:),12,2.1,1);
peak_locs = cat(2,peak_locs,temp_peak_locs);

[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,sound_fields,sound_field_ids);

[pupil_conv,temp_peak_locs] = conv_any_signal_final(pupil,-4,0,1,8);
% peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,pupil_conv,'pupil');
[pupil_conv,temp_peak_locs] = conv_any_signal_final(pupil,0,4,1,8);
% peak_locs = cat(2,peak_locs,temp_peak_locs);
[big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,pupil_conv,'pupil');



for i = 1:size(big_matrix,1);
    big_matrix(i,:) = big_matrix(i,:)./max(big_matrix(i,:));
end
temp_features_used = zeros(1,size(the_matrix,1));
for f = 1:size(the_matrix,1);
    for i = 1:length(big_matrix_ids);
        if length(big_matrix_ids{i})==length(the_matrix_ids{f})
                if big_matrix_ids{i}==the_matrix_ids{f}
                    temp_features_used(f) = 1;
                else
                    
                end
        end
    end
end

% the_matrix_used = the_matrix([(find(temp_features_used)) 33 34],:);
% the_matrix_ids_used = the_matrix_ids([(find(temp_features_used)) 33 34]);
the_matrix_used = the_matrix([(find(temp_features_used))],:);
the_matrix_ids_used = the_matrix_ids([(find(temp_features_used))]);

elseif type == 'cells';
    
    big_matrix = [];
    big_matrix_ids = {};
    numfilts = 10;
    for cel = 1:size(response,1);
%         response_conv = conv_any_signal_v30dd(response(cel,:)./max(response(cel,:)),-10,0,3,numfilts);
%         [big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,response_conv,cel);
        [response_conv,peak_locs] = conv_any_signal_final(response(cel,:)./max(response(cel,:)),0,10,3,numfilts);
%         filters_to_keep = 1:5;
        [big_matrix,big_matrix_ids] = add_filtered_features(big_matrix,big_matrix_ids,response_conv(filters_to_keep,:),cel);
        the_matrix_used(cel,:) = response(cel,:);
        the_matrix_ids_used(cel) = {cel};
        if length(find(isnan(big_matrix)))>0
            big_matrix(find(isnan(big_matrix)))=0;
        end
            
    end
    
    
end

big_matrix(find(isnan(big_matrix)))=0;