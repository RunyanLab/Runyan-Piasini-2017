function [upcoming_left_space,upcoming_right_space] = combine_space_turn(upcoming_left,upcoming_right,y_fields);

lr = upcoming_left - upcoming_right;

upcoming_left_space = zeros(size(y_fields));
upcoming_right_space = zeros(size(y_fields));

turns = find(lr);
if isempty(turns)==0
    t = 1;
    if lr(turns(t))>0;
        upcoming_left_space(:,1:turns(t)) = y_fields(:,1:turns(t));
    else
        upcoming_right_space(:,1:turns(t)) = y_fields(:,1:turns(t));
    end
    
    for t = 2:length(turns);
        if lr(turns(t))>0;
            upcoming_left_space(:,turns(t-1):turns(t)) = y_fields(:,turns(t-1):turns(t));
        else
            upcoming_right_space(:,turns(t-1):turns(t)) = y_fields(:,turns(t-1):turns(t));
        end
    end
end