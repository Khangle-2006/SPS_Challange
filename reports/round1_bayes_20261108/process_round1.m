function process_round1(videoPath, outputPath, pythonEntryPoint, frozenConfig)
% ROUND1 packaging skeleton, not a frozen submission.
% Official document names the input/output files but supplies no exact CSV columns.
% pythonEntryPoint must be the parity-verified portable exporter selected after G2.
% No reference PR is passed to inference. No BTC dataset-specific branching.
arguments
    videoPath (1,1) string = "target_video.mp4"
    outputPath (1,1) string = "processed_output.csv"
    pythonEntryPoint (1,1) string = "process_round1.py"
    frozenConfig (1,1) string = "frozen_configuration.json"
end
assert(isfile(videoPath), "Input video does not exist.");
assert(isfile(frozenConfig), "No frozen configuration. This is not a release.");
assert(isfile(pythonEntryPoint), "Portable Python exporter not yet packaged.");
assert(~isfile(outputPath), "Preserve existing output; choose a new destination.");
% pyrunfile passes paths as variables, avoiding shell interpolation and credentials.
pyrunfile(pythonEntryPoint, video_path=videoPath, output_path=outputPath, ...
    configuration_path=frozenConfig);
assert(isfile(outputPath), "Exporter did not produce an output file.");
end
