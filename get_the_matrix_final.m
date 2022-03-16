function [response,the_matrix,the_matrix_ids,this_trial_start,thisloc,thisfreq] = get_the_matrix_v12(imaging);
this_trial_start = zeros(1,length(imaging));
% if type=='behav'
    
    %%need to find a way to get rid of the bad values in "the_matrix"
    %%%%get the matrix of features
    %
    
    y_position = [];
    
    x_velocity = [];
    y_velocity = [];
    view_angle = [];
    reward = [];
    noreward = [];
    %     non_reward = [];
    
    %     trial_start = [];
    frequency_list = [];
    for v = 1:length(imaging)
        if isempty(imaging(v).binned)==0
            if min(imaging(v).sound_freq(1:19)=='ripple_snippet_pair')==1
                if ismember(str2num(imaging(v).sound_freq(21:end))+200,frequency_list)==0;
                    frequency_list(length(frequency_list)+1) = str2num(imaging(v).sound_freq(21:end))+200;
                    
                end
                %                 imaging(v).f_cat =  str2num(imaging(v).sound_freq(21:end))+200;
                imaging(v).f_cat = find(frequency_list==str2num(imaging(v).sound_freq(21:end))+200);
            elseif min(imaging(v).sound_freq(1:18)== 'ripple_snippet_sum')==1
                if ismember(str2num(imaging(v).sound_freq(20:end))+100,frequency_list)==0;
                    frequency_list(length(frequency_list)+1) = str2num(imaging(v).sound_freq(20:end))+100;
                    
                end
                %                 imaging(v).f_cat = str2num(imaging(v).sound_freq(20:end))+100;
                imaging(v).f_cat = find(frequency_list==str2num(imaging(v).sound_freq(20:end))+100);
            end
        end
    end
    previous_trial_correct = [];
    previous_trial_condition = [];
    %     is_reward = [];
    %     initi = [];
    sound_loc_list = [1 5 3 7 8 4 6 2];
    locs = [-90 -60 -30 -15 15 30 60 90];
    loc_stim1 = [];
    loc_stim2 = [];
    loc_stim3 = [];
    freq_stim = [];
    sound_on = [];
    upcoming_turn = [];
    
    response = [];
    
    for v_trial = 1:length(imaging)
        if isempty(imaging(v_trial).binned)==0
            
            this_y_position = zeros(1,size(imaging(v_trial).dff_zscored,2));
            
            this_x_velocity = zeros(1,size(imaging(v_trial).dff_zscored,2));
            this_y_velocity = zeros(1,size(imaging(v_trial).dff_zscored,2));
            this_view_angle = zeros(1,size(imaging(v_trial).dff_zscored,2));
            this_upcoming_turn = zeros(2,size(imaging(v_trial).dff_zscored,2));
            for i = 1:size(imaging(v_trial).dff_zscored,2);
                ind = find(imaging(v_trial).frame_id-min(imaging(v_trial).frame_id)+1==i,1,'first');
                if isempty(ind)==0
                    this_y_position(i) = imaging(v_trial).y_position(ind);
                    
                    this_x_velocity(i) = imaging(v_trial).x_velocity(ind);
                    this_y_velocity(i) = imaging(v_trial).y_velocity(ind);
                    this_view_angle(i) = imaging(v_trial).view_angle(ind);
                    
                else
                    this_y_position(i) = NaN;
                    
                    this_x_velocity(i) = NaN;
                    this_y_velocity(i) = NaN;
                    this_view_angle(i) = NaN;
                end
            end
            
            [bad_vals] = find(isnan(this_y_position));
            
            for bv = 1:length(bad_vals);
                this_y_position(bad_vals(bv)) = mean([this_y_position(bad_vals(bv)-1) this_y_position(bad_vals(bv)+1)]);
                this_y_velocity(bad_vals(bv)) = mean([this_y_velocity(bad_vals(bv)-1) this_y_velocity(bad_vals(bv)+1)]);
                this_x_velocity(bad_vals(bv)) = mean([this_x_velocity(bad_vals(bv)-1) this_x_velocity(bad_vals(bv)+1)]);
                this_view_angle(bad_vals(bv)) = mean([this_view_angle(bad_vals(bv)-1) this_view_angle(bad_vals(bv)+1)]);
                if isnan(this_y_position(bad_vals(bv)))==1
                    i = find(isnan(this_y_position(1:bad_vals(bv))-1)==0,1,'last');
                    this_y_position(bad_vals(bv)) = this_y_position(i);
                    this_y_velocity(bad_vals(bv)) =  this_y_velocity(i);
                    this_x_velocity(bad_vals(bv)) = this_x_velocity(i);
                    this_view_angle(bad_vals(bv)) = this_view_angle(i);
                end
            end
            
            this_trial_start(v_trial) = size(y_position,2)+1;
            y_position = cat(2,y_position,this_y_position);
            
            x_velocity = cat(2,x_velocity,this_x_velocity);
            y_velocity = cat(2,y_velocity,this_y_velocity);
            view_angle = cat(2,view_angle,this_view_angle);
            temp_loc_stim_r1 = zeros(1,size(imaging(v_trial).dff_zscored,2));
            temp_loc_stim_r2 = zeros(1,size(imaging(v_trial).dff_zscored,2));
            temp_loc_stim_r3andup = zeros(1,size(imaging(v_trial).dff_zscored,2));
            this_loc_stim1 = zeros(length(locs),size(imaging(v_trial).dff_zscored,2));
            this_loc_stim2 = zeros(length(locs),size(imaging(v_trial).dff_zscored,2));
            this_loc_stim3 = zeros(length(locs),size(imaging(v_trial).dff_zscored,2));
            this_freq_stim = zeros(length(frequency_list),size(imaging(v_trial).dff_zscored,2));
            this_sound_on = zeros(1,size(imaging(v_trial).dff_zscored,2));
            this_reward = zeros(1,size(imaging(v_trial).dff_zscored,2));
            this_noreward = zeros(1,size(imaging(v_trial).dff_zscored,2));
            x = find(sound_loc_list ==imaging(v_trial).sound_location);
            
            %             if imaging(v_trial).correct==1;
            %             reward_frame = imaging(v_trial).reward_frame_id - imaging(v_trial).frame_id(1);
            reward_frame = imaging(v_trial).frame_timing_info.reward_frames;
            %                 for bin = 1:length(imaging(v_trial).binned.frame_ids);
            %                     if ismember(reward_frame,imaging(v_trial).binned.frame_ids{bin})==1;
            if imaging(v_trial).correct==1
                this_reward(reward_frame) = 1;
            else
                this_noreward(reward_frame) = 1;
            end
            %                     end
            %                 end
            %             end
            
            reward = cat(2,reward,this_reward);
            noreward = cat(2,noreward,this_noreward);
            location = locs(x);
            thisloc(v_trial) = locs(x);
            thisfreq(v_trial) = imaging(v_trial).f_cat;
            %             for i = 1:length(imaging(v_trial).frame_timing_info.stimulus_on_frames)
            %                 this_sound_on(imaging(v_trial).frame_timing_info.stimulus_on_frames(i):imaging(v_trial).frame_timing_info.stimulus_end_frames(i)) = 1;
            frame_timing = imaging(v_trial).frame_timing_info.stimulus_on_frames;
            if max(diff(frame_timing))<25
                frame_timing = frame_timing(1:2:end);
            end
            
            temp_loc_stim_r1(frame_timing(1)) = 1;
            if length(frame_timing)>1
                temp_loc_stim_r2(frame_timing(2)) = 1;
            end
            if length(frame_timing)>2
                for i = 3:length(frame_timing)
                    temp_loc_stim_r3andup(frame_timing(i)) = 1;
                    
                    %                 this_freq_stim(imaging(v_trial).f_cat,imaging(v_trial).frame_timing_info.stimulus_on_frames(i):imaging(v_trial).frame_timing_info.stimulus_end_frames(i)) = 1;
                end
            end
            %             temp_loc_stim(imaging(v_trial).frame_timing_info.stimulus_on_frames(1):imaging(v_trial).frame_timing_info.stimulus_end_frames(length(imaging(v_trial).frame_timing_info.stimulus_on_frames))) = 1;
            %             this_freq_stim(imaging(v_trial).f_cat,imaging(v_trial).frame_timing_info.stimulus_on_frames(1):imaging(v_trial).frame_timing_info.stimulus_end_frames(length(imaging(v_trial).frame_timing_info.stimulus_on_frames))) = 1;
            
            
            
            loc_gauss = gauss_car([0 1 locs(x) 15],locs);
            loc_gauss = zeros(1,length(temp_loc_stim_r1));
            loc_gauss(x) = 1;
            for l = 1:length(locs);
                this_loc_stim1(l,:) = temp_loc_stim_r1.*loc_gauss(l);
                this_loc_stim2(l,:) = temp_loc_stim_r2.*loc_gauss(l);
                this_loc_stim3(l,:) = temp_loc_stim_r3andup.*loc_gauss(l);
            end
            
            loc_stim1 = cat(2,loc_stim1,this_loc_stim1);
            loc_stim2 = cat(2,loc_stim2,this_loc_stim2);
            loc_stim3 = cat(2,loc_stim3,this_loc_stim3);
            %             freq_stim = cat(2,freq_stim,this_freq_stim);
            %             sound_on = cat(2,sound_on,this_sound_on);
            
            if imaging(v_trial).leftTurn==1;
                %                 this_upcoming_turn(1,:) = ones(1,size(imaging(v_trial).dff_zscored,2)).*linspace(0,1,size(imaging(v_trial).dff_zscored,2));
                %                   a = find(abs(imaging(v_trial).binned.x_position)>1,1,'first');
                a = find(abs(imaging(v_trial).x_position)>1,1,'first');
                id = imaging(v_trial).frame_id(a)-imaging(v_trial).frame_id(1);
                this_upcoming_turn(1,id) = 1;
            else
                %                 this_upcoming_turn(2,:) = ones(1,size(imaging(v_trial).dff_zscored,2)).*linspace(0,1,size(imaging(v_trial).dff_zscored,2));
                %                 a = find(abs(imaging(v_trial).binned.x_position)>1,1,'first');
                a = find(abs(imaging(v_trial).x_position)>1,1,'first');
                id = imaging(v_trial).frame_id(a)-imaging(v_trial).frame_id(1);
                this_upcoming_turn(2,id) = 1;
            end
            upcoming_turn = cat(2,upcoming_turn,this_upcoming_turn);
            %         correct = ones(1,size(imaging(v_trial).dff_zscored,2)).*imaging(v_trial).leftTurn;
            
            response = cat(2,response,imaging(v_trial).dff_zscored);
            %             trial_start = cat(2,this_trial_start,trial_start);
        end
    end
    
    
    %     the_matrix = [y_position;x_position;view_angle;x_velocity;y_velocity;is_reward;initi;sound_on;...
    %         loc_stim; freq_stim; time; upcoming_turn];
    the_matrix = [y_position;y_velocity;x_velocity;view_angle;upcoming_turn;reward;noreward;...
        loc_stim1; loc_stim2; loc_stim3];
    the_matrix_ids{1} = 'y position';
    the_matrix_ids{2} = 'y velocity';
    the_matrix_ids{3} = 'x velocity';
    the_matrix_ids{4} = 'view angle';
    the_matrix_ids{5} = 'upcoming left turn';
    the_matrix_ids{6} = 'upcoming right turn';
    the_matrix_ids{7} = 'reward';
    the_matrix_ids{8} = 'noreward';
    the_matrix_ids{9} = 'location -90';
    the_matrix_ids{10} = 'location -60';
    the_matrix_ids{11} = 'location -30';
    the_matrix_ids{12} = 'location -15';
    the_matrix_ids{13} = 'location 15';
    the_matrix_ids{14} = 'location 30';
    the_matrix_ids{15} = 'location 60';
    the_matrix_ids{16} = 'location 90';
    the_matrix_ids{17} = 'location -90';
    the_matrix_ids{18} = 'location -60';
    the_matrix_ids{19} = 'location -30';
    the_matrix_ids{20} = 'location -15';
    the_matrix_ids{21} = 'location 15';
    the_matrix_ids{22} = 'location 30';
    the_matrix_ids{23} = 'location 60';
    the_matrix_ids{24} = 'location 90';
    the_matrix_ids{25} = 'location -90';
    the_matrix_ids{26} = 'location -60';
    the_matrix_ids{27} = 'location -30';
    the_matrix_ids{28} = 'location -15';
    the_matrix_ids{29} = 'location 15';
    the_matrix_ids{30} = 'location 30';
    the_matrix_ids{31} = 'location 60';
    the_matrix_ids{32} = 'location 90';
% end