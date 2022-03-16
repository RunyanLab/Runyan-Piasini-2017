function [new_signals,pk_loc] = conv_any_signal_v6(signal,pre_time,post_time,method,numfilts);
%%convolve a feature (x-variable) with a set of basis functions - defining
%%the pre-time and post-time (must include 0, so either a prespective or
%%retrospective filter.  ex: pre_time = -2, post_time = 0 for retro, or
%%pre_time = 0, post_time = 2 for pre.  Time is in seconds)  Define the
%%method for either linear or log spacing of the basis functions, as well
%%as the total number of filters.

%%method = 1 for linear
%%method = 2 for log
y = [];
duration = ceil((post_time-pre_time).*29.99);  %%15.63 is the imaging frame rate.
%%%define basis functions...
if method == 1;
    
    pk_loc = linspace(1,duration,numfilts+1);
    for b = 1:length(pk_loc)-1;
        hwhh = (pk_loc(b+1)-pk_loc(b));
        tw = hwhh/(2*sqrt(2*log(2)));
        bases(b,:) = gauss_car([0 1 pk_loc(b)  tw],1:duration);
        bases(b,:) = bases(b,:)./max(bases(b,:));
    end
    if pre_time==0;
    else
        bases = fliplr(bases);
        
    end
elseif method == 2;
    pk_loc = logspace(0,log10(duration),numfilts+1);
    
    for b = 1:length(pk_loc)-1;
        hwhh = (pk_loc(b+1)-pk_loc(b));
        tw = hwhh/(2.75*sqrt(2*log(2)));
        bases(b,:) = gauss_car([0 1 pk_loc(b) tw],1:duration);
        bases(b,:) = bases(b,:)./max(bases(b,:));
    end
    if pre_time==0;
    else
        bases = fliplr(bases);
        
    end
    
elseif method == 3;
    pk_loc = logspace(0,log10(duration),numfilts+1);
    for b = 2:length(pk_loc);
        if pk_loc(b)-pk_loc(b-1)<3;
            pk_loc(b)=pk_loc(b-1)+3;
        end
    end
    pk_loc = pk_loc(find(pk_loc));
    
    for b = 1:length(pk_loc)-1;
        hwhh = (pk_loc(b+1)-pk_loc(b));
        tw = hwhh/(2*sqrt(2*log(2)));
        bases(b,:) = gauss_car([0 1 pk_loc(b) tw],1:duration);
        bases(b,:) = bases(b,:)./max(bases(b,:));
    end
    if pre_time==0;
    else
        bases = fliplr(bases);
        
    end
    
elseif method == 4;
    pk_loc = logspace(0,log10(duration),numfilts+1);
    xdata = logspace(1,log10(duration),duration);
     for b = 1:length(pk_loc)-1;
        hwhh = (pk_loc(b+1)-pk_loc(b));
        tw = hwhh/(2.75*sqrt(2*log(2)));
        
        bases(b,:) = gauss_car([0 1 pk_loc(b) tw],1:duration);
        bases(b,:) = bases(b,:)./max(bases(b,:));
    end
    if pre_time==0;
    else
        bases = fliplr(bases);
        
    end
    
end
%
% figure(99)
% plot(bases')
extra_length = size(bases,2);

%%pad the signal with zeros prior to convolution, to lessen edge effects
signal_endsreflected = cat(2,squeeze(signal(:,extra_length:-1:1)),squeeze(signal(:,:)),squeeze(signal(:,end:-1:end-extra_length+1)));

ind = 0;

%%convolve x with the basis functions
for b = 1:size(bases,1);
    ind = ind+1;
    new_signal(ind,:) = conv(signal_endsreflected,bases(b,:),'same');
    new_signal(ind,:) = new_signal(ind,:)./max(new_signal(ind,:));
end

%%chop off the pad, with the position depending on pre/post time
%%relationship
if abs(pre_time)>0
    if abs(post_time)>0;
        if abs(pre_time)>abs(post_time)
            shift = abs((post_time/pre_time));
            s1 = 1.5+shift/2;
            s2 = 2-s1
        elseif abs(pre_time)<abs(post_time)
            shift = abs(pre_time/post_time);
            s2 = 1.5+shift/2;
            s1 = 2-s1;
            
            
        elseif abs(post_time) == abs(pre_time);
            
            s1 = 1;
            s2 = 1;
        end
    end
else
    s1 = 0.5;
    s2 = 1.5;
    new_signal2 = new_signal(:,(1+extra_length*s1:end-extra_length*s2));
end
if post_time==0;
    %     s1 = 1.45;
    %     s2 = 0.55;
    s1 = 1.5;
    s2 = .5;
    new_signal2 = new_signal(:,(1+extra_length*s1:end-extra_length*s2));
end


new_signals = new_signal2;

