function process_round1(roiPath, outputPrefix)
% Toolbox-free POS/Welch packaging baseline. NOT the selected BTC estimator.
% Input MAT: roi_data.t (seconds), rgb (N x 3), valid (N x 1).
% ROI extraction is precomputed and hash-bound by the exporter. No video reader.
% No labels are used for inference. Not yet validated in real MATLAB.
if nargin < 1, roiPath = 'roi_data.mat'; end
if nargin < 2, outputPrefix = 'processed_output'; end
assert(exist(roiPath, 'file') == 2, 'Missing ROI input');
assert(exist([outputPrefix '.csv'], 'file') ~= 2, 'Output exists; preserve it');
a = load(roiPath); assert(isfield(a, 'roi_data'), 'Missing roi_data');
roi_data = a.roi_data; t = double(roi_data.t(:)); rgb = double(roi_data.rgb);
valid = logical(roi_data.valid(:));
assert(size(rgb,1) == numel(t) && size(rgb,2) == 3, 'RGB shape');
assert(numel(valid) == numel(t) && all(diff(t) > 0), 'Clock shape/order');
params = struct('fs',30,'window_seconds',10,'hop_seconds',10, ...
    'POS_samples',48,'welch_samples',128,'welch_overlap',64,'nfft',4096, ...
    'band_hz',[0.7 3.0],'method','toolbox_free_POS_Welch_baseline', ...
    'MATLAB_verified',false,'selected_BTC_method',false);
starts = 0:10:(t(end)-10+1e-8); result = zeros(numel(starts),4);
processed_signal = cell(numel(starts),1);
for k = 1:numel(starts)
    start = starts(k); q = start+(0:299)'/30; ending = start+10;
    j0 = find(t <= start,1,'last'); j1 = find(t >= ending,1,'first');
    ok = ~isempty(j0) && ~isempty(j1);
    if ok, ok = all(valid(j0:j1)) && all(diff(t(j0:j1)) <= 0.2+1e-8); end
    hr = NaN;
    if ok
        x = interp1(t,rgb,q,'linear'); w = local_pos(x,48);
        processed_signal{k} = [q w];
        if all(isfinite(w)) && std(w,0) >= 1e-10
            hr = local_welch_hr(w,30,128,64,4096);
        end
    end
    result(k,:) = [start ending hr double(isfinite(hr))];
end
accepted = result(isfinite(result(:,3)),3);
metrics = struct('candidate_windows',size(result,1),'valid_windows',numel(accepted),'video_PR_bpm',NaN);
if ~isempty(accepted), metrics.video_PR_bpm = median(accepted); end
reference_signal = []; facial_video = '';
if isfield(a,'facial_video'), facial_video = a.facial_video; end
if isfield(a,'reference_signal'), reference_signal = a.reference_signal; end
fid = fopen([outputPrefix '.csv'],'w'); assert(fid >= 0,'Cannot open CSV');
fprintf(fid,'window_start_s,window_end_s,PR_bpm,valid\n'); fclose(fid);
dlmwrite([outputPrefix '.csv'],result,'-append','delimiter',',','precision','%.12g');
fid = fopen([outputPrefix '_summary.csv'],'w'); assert(fid >= 0,'Cannot open summary');
fprintf(fid,'video_PR_bpm,valid_windows,candidate_windows\n');
fprintf(fid,'%.12g,%d,%d\n',metrics.video_PR_bpm,metrics.valid_windows,metrics.candidate_windows); fclose(fid);
save([outputPrefix '.mat'],'facial_video','roi_data','reference_signal','processed_signal','metrics','params','-v7');
end

function h = local_pos(x,L)
n = size(x,1); h = zeros(n,1);
for last = L:n
    idx = (last-L+1):last; block = x(idx,:); m = mean(block,1);
    if any(abs(m) < 1e-10), continue; end
    c = bsxfun(@rdivide,block,m)'; s = [0 1 -1;-2 1 1]*c;
    denom = std(s(2,:),0); if denom < 1e-10, continue; end
    z = s(1,:)+std(s(1,:),0)/denom*s(2,:); z = z-mean(z);
    h(idx) = h(idx)+z';
end
end

function hr = local_welch_hr(w,fs,L,overlap,nfft)
n = numel(w); tt = linspace(-1,1,n)'; w = w-mean(w);
w = w-(tt'*w)/(tt'*tt)*tt;
window = 0.5-0.5*cos(2*pi*(0:L-1)'/L); power = zeros(nfft/2+1,1); count = 0;
for start = 1:(L-overlap):(n-L+1)
    z = w(start:start+L-1); z = z-mean(z); y = fft(z.*window,nfft);
    power = power+abs(y(1:nfft/2+1)).^2; count = count+1;
end
power = power/max(count,1); f = (0:nfft/2)'*fs/nfft;
sel = find(f >= 0.7 & f <= 3); hr = NaN;
if all(isfinite(power)) && sum(power(sel)) > 1e-30
    [unused,index] = max(power(sel)); hr = 60*f(sel(index));
end
end
