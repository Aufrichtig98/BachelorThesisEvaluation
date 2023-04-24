from varats.report.tef_report import TEFReport, WorkloadSpecificTEFReportAggregate

from pathlib import Path
from os import mkdir

from varats.experiment.experiment_util import (
    ExperimentHandle,
    get_varats_result_folder,
    VersionExperiment,
    get_default_compile_error_wrapped,
    get_varats_result_folder, ZippedReportFolder
)
from tempfile import TemporaryDirectory

def parse_files_into_tef(file_path:Path, result_path:Path):
    number_of_files = 0
    for zip in file_path.iterdir():
        print(zip)
        tef_reports = WorkloadSpecificTEFReportAggregate(zip)
        print("Read out")
        file_name = zip.stem
        dir_path = result_path / file_name
        mkdir(dir_path)
        
        for keys in tef_reports.keys():
            for reports in tef_reports.reports(keys):
                reports.feature_time_accumulator(dir_path / (keys + ".json"))
                print(reports)
                
        TEFReport.wall_clock_times(dir_path, file_name)
        
if __name__ == '__main__':

    file_path = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/resultsGroundTruth/results/FPR-TEF-FeaturePerfCSCollection-GTBasic-144cd3143d")
    result_path = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/resultsGroundTruth/resultsEval/GTBasic")
    
    #file_path = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/testResults/FPR-TEF-FeaturePerfCSCollection-GTSharedFeature-144cd3143d")
    #result_path = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/testAccu")
    
    parse_files_into_tef(file_path, result_path)