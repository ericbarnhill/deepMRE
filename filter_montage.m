function filter_montage(filters)
    nf = numel(filters);
    for i = 1:nf
        sz2 = size(filters{i});
        for j = 1:sz2(1)
            ind = (i-1)*sz2(1) + j;
            disp(ind)
            subplot(nf, sz2(1), ind);
            imshow(squeeze(filters{i}(j,:,:)), []);
        end
    end
impixelinfo