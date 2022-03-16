function [y_fields,y_field_ids] = make_place_fields_v28(the_matrix);
y_data = the_matrix(1,:);
maze_end = round(max(y_data)/100)*100;
place_field_centers = linspace(25,maze_end,30);
fields = zeros(length(place_field_centers),maze_end);
tw = mean(diff(place_field_centers));

for pf = 1:length(place_field_centers)
    fields(pf,:) = gauss_car([0 1 place_field_centers(pf) tw],1:maze_end);
end


y_fields = zeros(size(fields,1),length(y_data));
for pf = 1:length(place_field_centers)
    for pts = 1:length(y_data);
        if y_data(pts)<=0
            y_fields(pf,pts) = 0;
        elseif y_data(pts)>maze_end;
            y_fields(pf,pts) = fields(pf,maze_end);
        else
            y_fields(pf,pts) = y_data(pts)*fields(pf,ceil(y_data(pts)));
        end
    end
end

for pf = 1:size(y_fields,1);
    y_fields(pf,:) = y_fields(pf,:)./max(y_fields(pf,:));
%     y_field_ids{pf} = ['Place Field ' num2str(pf)];
    y_field_ids{pf} = 'y position';
end
y_fields(pf+1,find(sum(y_fields)<0.01)) = 1;
% y_field_ids{pf+1} = ['Nowhere Place Field'];
y_field_ids{pf+1} = 'y position';