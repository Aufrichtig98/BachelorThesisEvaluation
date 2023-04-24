from pathlib import Path
import json
import pandas as pd
from functools import cmp_to_key
import pandas as pd
from sklearn.linear_model import LinearRegression


def aggregate_all_results(path:Path) -> dict:
    selected_feature = list()

    for config_folder in path.iterdir():
        configuration_options = [x[-1] for x in str(config_folder).split("/")[-1].split("_")[1:]]
        json_result_path = config_folder \
            / f"XZ_comp{configuration_options[0]}_extreme{configuration_options[1]}_threads{configuration_options[2]}.json"
        
        f = open(json_result_path)
        json_file = json.load(f)
        for features in json_file:
            if features == "Overall time for all features":
                break
            if json_file[features]["Overall Time"] > 1.0:
                if features not in selected_feature:
                    selected_feature.append(features)
    print(selected_feature)
    #Unefficient to repeat this but date we work with is neglible small
    
    row_names = list()
    #List with feature_measurements[0] = name of selected features
    features_measurements = list() 
    config_to_features = dict()
    for config_folder in path.iterdir():
        configuration_options = [x[-1] for x in str(config_folder).split("/")[-1].split("_")[1:]]
        json_result_path = config_folder \
            / f"XZ_comp{configuration_options[0]}_extreme{configuration_options[1]}_threads{configuration_options[2]}.json"
        
        f = open(json_result_path)
        json_file = json.load(f)

        config_name = "_".join(configuration_options)
        measurements_current_config = list()
        
        for features in selected_feature:
            measurements_current_config.append(json_file[features]["Overall Time"])
        
        config_to_features[config_name] = measurements_current_config

    row_names = list()
    for comp in range(10):
        for extreme in range(2):
            for threads in range(5):
                if threads:
                    name = f"{comp}_{extreme}_{1 << (threads-1)}"
                else:
                    name = f"{comp}_{extreme}_{threads}"
                row_names.append(name)
                features_measurements.append(config_to_features[name])

    result_aggregated = pd.DataFrame(data=features_measurements, columns=selected_feature, index=row_names)
    print(result_aggregated)
    perf_model_feature_xz(config_to_features)
    
def perf_model_feature_xz(predict_to_time):
    

    config = list(predict_to_time.keys())
    times = list(predict_to_time.values())
    
    base_time = list()
    base_features = list()
    for i in range(len(times)):
        base_time.append(times[i][0])
        base_features.append([1])
    threads_time = list()
    threads_features = list()
    
    #fer each threats feature
    for i in range(5):
        thread_lvl = 0
        tmp_threads_time = list()
        tmp_threads_feature = list()
        if i:
            thread_lvl = (1 << (i-1))
        for j in range(len(times)):
        
            #checks the last char of config to see if the current thread is on
            if str(thread_lvl) == config[j][-1]:
                tmp_threads_time.append(times[j][1])
                tmp_threads_feature.append([1])
            else:
                tmp_threads_time.append(0)
                tmp_threads_feature.append([0])
        threads_time.append(tmp_threads_time)
        threads_features.append(tmp_threads_feature)
     
    result_times = [base_time] + threads_time
    result_features = [base_features] + threads_features
    
    for i in range(len(result_times)):
        regression = LinearRegression().fit(result_features[i], result_times[i])
        print(round((regression.intercept_ / 1000),ndigits=3))
        print(round((regression.coef_[0] / 1000), ndigits=3))

def aggregate_all_results_groundTruth(path:Path) -> dict:
    selected_feature = list()

    for ground_truth in path.iterdir():
        system_name = (str(ground_truth)).split("/")[-1]

        selected_feature = ['Base', 'Base,FeatureA', 'Base,FeatureB', 'Base,FeatureC', 'Base,FeatureD', 
                            'Base,FeatureA,FeatureB' ,'Base,FeatureC,FeatureD']
        if "GTShared" in str(ground_truth):
            selected_feature = ['Base', 'Base,FeatureA', 'Base,FeatureB', 'Base,FeatureC', 'Base,FeatureD', 'Base,FeatureE',
                                'Base,FeatureA,FeatureB' ,'Base,FeatureC,FeatureD']
        #Unefficient to repeat this but date we work with is neglible small
        
        row_names = list()
        #List with feature_measurements[0] = name of selected features
        features_measurements = list() 
        config_to_features = dict()
        for config_folder in ground_truth.iterdir():
            configuration_options = [x for x in str(config_folder).split("/")[-1].split("_")[1:]]
            feature_folder_name = "trace"
            for feature in configuration_options:
                feature_folder_name = feature_folder_name + f"_{feature}"
            feature_folder_name = feature_folder_name + ".json"
            json_result_path = config_folder / feature_folder_name
            
            f = open(json_result_path)
            json_file = json.load(f)

            config_name = "_".join(configuration_options)
            measurements_current_config = list()

            for features in selected_feature:
                if features in json_file:                    
                    measurements_current_config.append(round((json_file[features]["Overall Time"])/1000, 3))
                else:
                    measurements_current_config.append(0)
                    
            config_to_features[config_name] = measurements_current_config
            row_names.append(config_name)
        
        row_names.sort()
        for name in row_names:
            features_measurements.append(config_to_features[name])
        for i in range(len(row_names)):
            row_names[i] = row_names[i].replace("base_", "").replace("base", "").replace("_", ", ").upper()
            row_names[i] = "ReplaceCurlyLeft" + row_names[i] + "ReplaceCurlyRight"
        column_names = list()
        for i in range(len(selected_feature)):
            column_names.append(selected_feature[i].replace("Base,", "").replace(",","land").replace("Feature", ""))
            
        result_aggregated = pd.DataFrame(data=features_measurements, columns=column_names, index=row_names).to_latex()
        result_aggregated = result_aggregated.replace("land"," $\land$ ").replace("ReplaceCurlyLeft", "\{").replace("ReplaceCurlyRight", "\}")
        
        #Calculates the perf model#
        if(system_name == "GTMulticollinearity"):
            print("nothing")
        perf_model = [0]
        for i in range(len(column_names)):
            current_feature = list()
            for j in range(len(features_measurements)):
                current_feature.append(features_measurements[j][i])
            regression = perf_model_feature(current_feature)
            if "GTElse" in str(ground_truth) and (i == 1):
                regression = perf_model_feature(current_feature, True)
            perf_model[0] += regression.intercept_
            if(i == 0):
                continue
            perf_model.append(regression.coef_[0])
        ###########################

        for i in range(len(perf_model)):
            perf_model[i] = round(perf_model[i],3)
        result_aggregated = result_aggregated + "\n\n"
        
        result_aggregated += pd.DataFrame(data=[perf_model], columns=column_names, index=[system_name]).to_latex()
        
        with open("/scratch/messerig/EvaluationScripts/ResultWhitebox/resultsGroundTruth/Latex/" + ground_truth.stem + ".txt", "w") as f:
            f.write(result_aggregated)

def perf_model_feature(predicted_time, else_case = False):
    feature_deactivated = 0
    if else_case:
        feature_deactivated = 2.0
    
    feature_active = list()
    for element in predicted_time:
        if element == feature_deactivated:
            feature_active.append([0])
        else:
            feature_active.append([1])
     
    return LinearRegression().fit(feature_active, predicted_time)
            


if __name__ == '__main__':
    result_folder_path = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/resultsAccumulated")
    result_folder_path_GroundTruth = Path("/scratch/messerig/EvaluationScripts/ResultWhitebox/resultsGroundTruth/resultsEval")
    aggregate_all_results(result_folder_path)
    #aggregate_all_results_groundTruth(result_folder_path_GroundTruth)