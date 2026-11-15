import json,sys,shlex
from pathlib import Path
r=Path(sys.argv[1]);text='# Submission-readiness check\n\n'
text+='The [official FaceVitals document](https://signalprocessingsociety.org/sites/default/files/2026-10/2027-SP-Cup-Official-Document-FaceVitals.pdf), pages10–11, names facial_video,roi_data,reference_signal,processed_signal,metrics,params. It does not fix detailed dtypes,axisorder,nestedstructlayout orMATversion. Thus technicalPASS below is distinct from officialcompatibilityUNKNOWN. The model uses the frozenlargeoval; roi_data carries separateT3 anatomicalpolygons.\n\n'
statuses={}
for ds,host in [('MCD','P100'),('UBFC2','A100')]:
    root=r/('receipts_'+host)/('packaging_'+ds)/'attempt2';a=root/'artifact'
    status=json.loads((a/'status.json').read_text());statuses[ds]=status
    text+='## '+ds+'\n\n'
    if (a/'FAILURE.json').exists():text+='**FAIL**: '+(a/'FAILURE.json').read_text()+'\n\n';continue
    assert status['state']=='complete'
    layout=json.loads((a/'FIELD_LAYOUT.json').read_text());par=json.loads((a/'OOF_PARITY.json').read_text());front=json.loads((a/'FRONTEND_PARITY.json').read_text());venv=json.loads((root/'FRESH_VENV.json').read_text())
    assert par['coverage_exact'] and par['reference_exact'] and front['all_pixels_exact'] and status['mat_verification']['required_fields_present'] and status['mat_verification']['rgb_hash_verified']
    text+=f"Record {status['recording']}; branch {status['branch']}, seed2027, heldoutfold{layout['params']['fold']}. No individualcheckpoint selection by score. Fullpipeline elapsed **{status['timings']['total_s']:.2f}s**.\n\n"
    text+='| Field | Actual type/shape | Technical | Official detail |\n|---|---|---|---|\n'
    image=layout['facial_video'];rois=layout['roi_data'];refs=layout['reference_signal'];pr=layout['processed_signal']
    text+=f"| facial_video | uint8 MATLAB{image['shape']} (H,W,RGB,T), embeddedv7.3 | PASS fullmat73readback + RGBhash | UNKNOWN axis/type/version requirements |\n"
    text+=f"| roi_data | MATLABstruct; forehead{rois['forehead']['shape']}, leftcheek{rois['cheek_left']['shape']}, rightcheek{rois['cheek_right']['shape']}; float32pixelpolygons,valid,timestamps | PASS trace/video timestamps exact | UNKNOWN nestedlayout |\n"
    text+=f"| reference_signal | MATLABstruct; float64PPG/time vectors length{refs['ppg_samples']}; canonicalpolarity; perwindowHR | PASS ownclock and HRparity | UNKNOWN waveformvsHR fieldlayout |\n"
    text+=f"| processed_signal | MATLABstruct; rPPG/time vectors length{pr['samples']}; {pr['windows']} HR/windows and abstainmask | PASS OOFmaxHRdelta{par['max_HR_delta_bpm']:.6f}bpm | UNKNOWN piecewisewindowed waveform acceptance |\n"
    text+='| metrics | MATLABstruct; MAE,RMSE,Pearson,bias,coverage; LoA=NA | PASS computation; LoA unresolved with1person | UNKNOWN exact metricfieldnames |\n'
    text+='| params | MATLABstruct; resolution,FPS,timestamps,polarity/checkpointprovenance; missingacquisitionmetadata=NA | PASS storedmetadata | UNKNOWN lux,distance,Fitzpatrick; no inventedvalues |\n\n'
    text+='Timing breakdown(seconds): `'+json.dumps(status['timings'])+'`. Frozenfrontend pixels/support/clocks are exact; decodedRGB matches both exportedMP4 and MAT. OOFreferenceHR and acceptedmask exact. CPUHRparity tolerance0.5bpm was fixed beforelaunch; actualdelta is reported above.\n\n'
    text+='Environment: freshvenvdirectory `'+venv['environment']+'`; Python/dependencyversions recorded in FRESH_VENV.json. Installeddependencies are exposed through explicit .pth paths, including systemdist-packages; **not an isolatedfresh dependency installation**.\n\n'
    params=layout['params'];hd=params['resolution'][0]>=1280 and params['resolution'][1]>=720
    text+=f"Acquisition: resolution{params['resolution']}, nominalFPS{params['fps_nominal']}, source-clockeffectiveFPS{params['fps_effective']:.6f}. HDdimensions: {'PASS' if hd else 'FAIL'}, assessed separately from serialization; lux/distance areUNKNOWN. Source-clockcadence is reported without claiming independentlyverified hardwaretiming.\n\n"
text+='## Remaining human items\n\n- Obtain organizerpilotdata and validate the exact organizerreader/schema; currenttechnicalPASS does not certify compatibility.\n- SupervisorConflict of Interestform.\n- Clarify reference_signalCSV: this package retains PPGCSV and separate reference_hrCSV; the document describes groundtruthHR, while waveformtraining usesPPG.\n- Confirm whether independent10s waveforms, v7.3H,W,3,T and nestedfields are accepted. No continuouswaveform has been fabricated.\n- Obtain qualifyingHD/acquisitionmetadata if required; no distance/lux/Fitzpatrick values are guessed.\n- Freshdependencyinstallation and a trueexternalvideo/no-label runtime adapter are not validated. These tests exercise datasetadapters for referencepacking and frozenreference-support policies; modelprediction itself does not consume PPGvalues.\n- MCDpolarity correction is global in the adapter; D3 is not opened.\n\nHistoricalsubmission/process_round1.py was not edited. These versionedCPU demonstrations are not a submission and do not change the evaluatedmodel.\n'
p=r/'schema_check.md';assert not p.exists();p.write_text(text)
(r/'PACKAGING_SUMMARY.json').write_text(json.dumps(statuses,indent=2));print('SCHEMA_REPORT_WRITTEN')
