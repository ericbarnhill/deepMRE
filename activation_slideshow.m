function activation_slideshow(y)

vec = @(x) x(:);

for n=1:size(y,1)
    slides = squeeze(y(n,:,:,:));
    sz = size(slides);
    disph1 = round(sz(1)*0.1);
    disph2 = round(sz(1)*0.65);
    dispw1 = round(sz(2)*0.1);    
    dispw2 = round(sz(2)*0.75);    
    fig = figure(1);
    if numel(sz) == 2
        d3 = 1;
    else
        d3 = sz(3);
    end
    for m = 1:d3
        slide_crop = squeeze(slides(disph1:disph2,dispw1:dispw2,m));
        im = imresize(slide_crop, 8);
        q1 = quantile(vec(slide_crop), 0.05);
        q2 = quantile(vec(slide_crop), 0.95);
        imshow(im, [q1,q2]);
        %truesize(fig, [sz(1)*8 sz(2)*8]);
        title(['Node ' num2str(n) ' filter ' num2str(m) ' of ' num2str(sz(1)) ...
            ' .05 ' num2str(q1) ' .95 ' num2str(q2)]);
        pause(1);
    end
end
    