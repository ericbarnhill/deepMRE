function filter_slideshow(y)

for n=1:numel(y)
    slides = y{n};
    sz = size(slides);
    fig = figure(1);
    for m = 1:sz(1)
        im = imresize(squeeze(slides(m,5:end-4,5:end-4)), 3);
        imshow(im, []);
        %truesize(fig, [sz(1)*8 sz(2)*8]);
        title(['Node ' num2str(n) ' filter ' num2str(m) ' of ' num2str(sz(1))]);
        pause(1);
    end
end
    