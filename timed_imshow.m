function timed_imshow(x)

sz = size(x);
for n = 6000:sz(1)
    imshow(squeeze(x(n,:,:)), []);
    pause(.01);
end