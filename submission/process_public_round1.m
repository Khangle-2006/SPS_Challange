function process_round1(repo_dir, output_dir, python_exe)
% Real MATLAB/Octave wrapper. Python performs the frozen CPU P4m computation.
% Place this file as Round1_Baseline/process_round1.m.
base = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(repo_dir), repo_dir = fullfile(base, '..', 'repo'); end
if nargin < 2 || isempty(output_dir), output_dir = fullfile(base, 'rerun'); end
if nargin < 3 || isempty(python_exe), python_exe = 'python3.12'; end
if exist(fullfile(repo_dir, 'predict_public.py'), 'file') ~= 2
    error('Clone SPS_Challange as ../repo, run tools/bootstrap_video.py and tools/setup_env.py torch-cpu, then supply repo_dir.');
end
script = fullfile(base, 'process_round1.py');
cmd = [quote_arg(python_exe) ' ' quote_arg(script) ' --bundle ' quote_arg(base) ' --repo ' quote_arg(repo_dir) ' --output ' quote_arg(output_dir)];
[status, text] = system(cmd);
fprintf('%s', text);
if status ~= 0
    error('P4m Python process failed. Check Python 3.12, ffmpeg, locked environments and release assets; inspect output logs.');
end
end

function q = quote_arg(s)
if any(s == char(10)) || any(s == char(13))
    error('Newlines are not valid command arguments.');
end
if ispc
    if any(s == char(34)) || any(s == '%'), error('Unsupported Windows command character.'); end
    q = [char(34) s char(34)];
else
    q = [char(39) strrep(s, char(39), [char(39) char(34) char(39) char(34) char(39)]) char(39)];
end
end
