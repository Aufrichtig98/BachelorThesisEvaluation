from varats.report.tef_report import TEFReport, WorkloadSpecificTEFReportAggregate

from pathlib import Path
from os import mkdir

def parse_files_into_tef(file_path:Path, result_path:Path):
    """
    file_path: Path to the folder that contains the zips with the measuremsts 
    result_path: Path to the folder where we want to put our results in
    
    Function reads in every zip in the folder conatining var trace reports, then aggregates the time spent in each region 
    and writes the overall time spent in each feature region in a file for the measured config
    
    """
    
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

    #file_path path to where the ZIPs with the trace files are located
    #result_path where we want to put the aggregated time in
    file_path = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/resultsGroundTruth/results/FPR-TEF-FeaturePerfCSCollection-GTBasic-144cd3143d")
    result_path = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/resultsGroundTruth/resultsEval/GTBasic")
    
    parse_files_into_tef(file_path, result_path)