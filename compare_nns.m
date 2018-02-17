%assemble predicted sets
load('predictions.mat');
models = {orig, E4_0, E8_0, E816_0};
n_mods = numel(models);

figure(1);
MIN=0.2;
rand_slice = round(rand*size(orig,1));
get_sl = @(x) squeeze(real(x(rand_slice,:,:)));
varname = @(x) inputname(1); %#ok<FINS>
get_min = @(x) min(x(x>MIN));
get_min_ = @(x) min(x(:));
get_max = @(x) max(x(:));
vec = @(x) x(:);

for n = 1:n_mods
    subplot(2, n_mods, n);
    title(varname(models{n}));
    if isempty(get_min(models{n}))
        mn = 0;
    else
        mn = get_min(models{n});
    end
    if get_max(models{n}) > 0
        imshow(get_sl(models{n}), [mn, get_max(models{n})]);
    end
    if n > 1
        subplot(2, n_mods, n + n_mods);
        resid = models{n} - models{1};
        imshow(get_sl(resid), [get_min_(resid), get_max(resid)]);
        title(num2str(norm(vec(resid))));
    end
end