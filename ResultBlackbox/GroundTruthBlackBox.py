import os
import numpy as py
from sklearn.linear_model import LinearRegression
from pathlib import Path
from varats.report.gnu_time_report import TimeReport, TimeReportAggregate
import pandas as pd
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy.highlevel import dmatrices
from typing import List, Dict
import statsmodels.api as sm


def feature_permutation(feature_list: List[str], legal:List[List[str]]) -> List[List[str]]:
    num_sets = 1 << len(feature_list)
    feature_permutation = []
    for current_mask in range(num_sets):

        string_list = []
        for i in range(len(feature_list)):
            if (current_mask & (1 << i)) != 0:
                string_list.append(feature_list[i])
        feature_permutation.append(string_list)

    for configuration in feature_permutation:
        for term in legal:
            if all([item] in configuration for item in term):
                configuration.append(["".join(term)])

    return feature_permutation
        
def feature_string_list(str_configuration: List[str], Shared=False):        
    result_config = [0] * 7
    if Shared:
        result_config = [0] * 8
    result_config[0] = 1
    for feature in str_configuration:
        if Shared:
            for i in range(6):
                if [chr(ord("a") + i)] == feature:
                    result_config[i + 1] = 1
                    break
            if feature == ["ab"]:
                result_config[6] = 1
                continue
            if feature == ["cd"]:
                result_config[7] = 1
        else:    
            for i in range(5):
                if [chr(ord("a") + i)] == feature:
                    result_config[i + 1] = 1
                    break
            if feature == ["ab"]:
                result_config[5] = 1
                continue
            if feature == ["cd"]:
                result_config[6] = 1

    return result_config
        

def feature_black_box(filename, Shared=False):
    if (Shared):
        result_list = [0] * 5
    else:
        result_list = [0] * 4
    filename = filename.split("/")[-1]
    
    if "--a" in filename:
        result_list[0] = 1
    if "--b" in filename:
        result_list[1] = 1
    if "--c" in filename:
        result_list[2] = 1
    if "--d" in filename:
        result_list[3] = 1      
    if Shared:
        if "--e" in filename:
            result_list[4] = 1  
    return result_list

    
def parse_zip_files(path:Path):
    #Used to meassure the mean of the time spend of for each zip file containing multiple time reports
    #path: Folder containing mutliple zips
    
    #Maps config (compression, extreme, threads) to time it took to run xz
    feature_config_to_time = dict()
    for zip in path.iterdir():
        if "Shared" in str(results_path):
            xz_features = feature_black_box(str(zip),True)
        else:
            xz_features = feature_black_box(str(zip))
        xz_result = TimeReportAggregate(zip)
        feature_config_to_time[tuple(xz_features)] = py.mean(xz_result.measurements_wall_clock_time)
    return feature_config_to_time

class multible_linear_reagression:
    
    @property
    def all_term(self):
        if "Shared" in str(results_path):
            feature_to_name = [["base"], ["a"], ["b"], ["c"], ["d"], ["e"],["a", "b"],["c", "d"]]   
        else:
            feature_to_name = [["base"], ["a"], ["b"], ["c"], ["d"], ["a", "b"],["c", "d"]]
        self.name_inverse = dict()
        self.name_inverse = {name[0]:i for i,name in enumerate(feature_to_name) if len(name)==1}
        return feature_to_name

    def __init__(self, feature_config_to_time:dict, path:Path):
        #Takes Dict containing feature selection mapping configuration -> time_spent
        
        #feature_name returns the feature/intearaction at given matrix pos
        
        self.path = path
        self.feature_name = self.all_term
        
        X = list()
        y = list()
        
        for config_time_pair in feature_config_to_time.items():
            
            config = config_time_pair[0]
            time_spent = config_time_pair[1]
            
            y.append(time_spent)
            
            if "Shared" in str(results_path):
                X.append(self.config_to_matrix(config,True))
            else:
                X.append(self.config_to_matrix(config))  
        
        
        names = [self.feature_name[i][0] for i in range(5)]
        if "Shared" in str(results_path):
            names = [self.feature_name[i][0] for i in range(6)]
            
        x_test = list()
        for elements in X:
            x_test.append([str(i) for i in elements])
        
        self.configurations = pd.DataFrame(x_test, columns=names, dtype=str)
        self.configurations['performance'] = pd.Series(y)
        
        filtered_config = self.apply_iterative_vif(self.feature_name, nfp="performance")
        
        regression_data = list()
        dependent_data_sorted = list()
        
        single_feature = list()
        feature_interactions = list()
        
        for config in filtered_config:
            if len(config) == 1:
                single_feature.append(config)
            else:
                feature_interactions.append(config)
                
    
        single_feature.remove(["base"])
        features = feature_permutation(single_feature, feature_interactions)
        if "Multi" in str(path):
            feature_filtered = list()
            for config in features:
                if ['b'] in config:
                    feature_filtered.append(config)
            features = feature_filtered
            
        for elements in features:
            
            if "Shared" in str(results_path):
                current_config = feature_string_list(elements,True)
            else:
                current_config = feature_string_list(elements)
            regression_data.append(current_config)
            if "Shared" in str(results_path):
                dependent_data_sorted.append(feature_config_to_time[tuple(current_config[1:6])])
                continue
            dependent_data_sorted.append(feature_config_to_time[tuple(current_config[1:5])])
    
           
        self.X = py.array(regression_data)
        self.y = py.array(dependent_data_sorted)
        self.linear_model = LinearRegression().fit(self.X, self.y)

        name_file = [str(path).split("/")[-1]]
        
        result_dataframe = self.linear_model.coef_
        result_dataframe[0] = self.linear_model.intercept_
        result_dataframe = list(result_dataframe)
        column_name = list()
        for config in filtered_config:
            if len(config) == 1:
                column_name.append(config[0])
            else:
                feature_name = config[0]
                config = config[1:]
                for feature in config:
                    feature_name = feature_name + f", {feature}"
                column_name.append(feature_name)
     
        if "Multi" in str(path):
            result_dataframe = result_dataframe[:5] + result_dataframe[6:]
       
        for i in range(len(result_dataframe)):
            result_dataframe[i] = round(result_dataframe[i],ndigits= 3)
        
        column_name_latex = list()
        column_name_latex.append("Base")
        for i in range(1,len(column_name)):
            column_name_latex.append(column_name[i].upper().replace(",","land"))
    
        result_table = pd.DataFrame(data=[result_dataframe], columns=column_name_latex, index=name_file).to_latex()
        result_table = result_table.replace("land", " $\land$")
        
        with open("/scratch/messerig/EvaluationScripts/ResultBlackbox/GroundTruth/Latex/" + name_file[0] + ".txt", "w") as f:
            f.write(result_table)

    #Method by Christian Kaltenecker#
    def apply_iterative_vif(self, model_to_check: List[List[str]], nfp: str, log_path: str = None, revision=None) -> List[List[str]]:
        """
        Applies an iterative VIF analysis. In each iteration, an additional term is included to the VIF analysis.
        Whenever the threshold if Infinity, the new term is removed. The removed term and the terms it is
        conflicting with is printed on the console.
        :param model_to_check: the model to check. It contains in each line a term of the performance-influence model
        :param nfp: the nfp to investigate
        :param log_path: the path to the log file where the conflicts are written
        :return: A reduced model where all conflicting model are already removed.
        """
        if len(model_to_check) < 2:
            print("The length of the given model is too short (less than 2)")
            exit(-1)
        log_file = None
        if log_path is not None:
            log_file = open(log_path, 'w')

        current_model = [model_to_check[0]]
        current_model_string = ['_'.join(model_to_check[0])]

        if revision is None:
            data = self.configurations
        else:
            data = self.configurations[self.configurations['revision'] == revision]

        

        dataframe_for_vif = pd.DataFrame(data=data, columns=[model_to_check[0][0], nfp])
        for i in range(1, len(model_to_check)):
            current_model.append(model_to_check[i])
            current_model_string.append('__'.join(model_to_check[i]))
            # Append to the dataframe the needed information
            # The term could consist of one or more terms; therefore, we have to split it
            dataframe_for_vif[current_model_string[-1]] = self.configurations[model_to_check[i][0]]
            for j in range(1, len(model_to_check[i])):
                dataframe_for_vif[current_model_string[-1]] = dataframe_for_vif[current_model_string[-1]].astype(int) \
                                                            * self.configurations[model_to_check[i][j]] \
                                                                .astype(int)
            # Construct the matrix for the VIF analysis
            y, X = dmatrices('performance ~ ' + '+'.join(current_model_string), data=dataframe_for_vif,
                            return_type='dataframe')
            # Run the analysis
            vif = pd.DataFrame()
            vif['variable'] = X.columns
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
            # Check if all VIFs are below infinity
            # If not, check with which terms the given term interferes.
            conflicting_terms = []
            vif_value = 0
            for index, row in vif.iterrows():
                # Values that are infinity mean that the variance is too high
                if (py.isinf(row['VIF']) or py.isnan(row['VIF'])) and row['variable'] != "Intercept":
                    conflicting_terms.append(row['variable'])
                    vif_value = row['VIF']
            if len(conflicting_terms) > 0:
                print(f"Removing term {current_model_string[-1]} since it is conflicting with {str(conflicting_terms)}")
                if log_file is not None:
                    log_file.write(f"{current_model_string[-1]} ({vif_value}): {str(conflicting_terms)}\n")
                current_model = current_model[:-1]
                current_model_string = current_model_string[:-1]
            # Note: There is currently no support for numeric features
        if log_file is not None:
            log_file.close()

        return current_model

    def config_to_matrix(self, config:tuple, Shared=False):
        
        feature_selection = [0] * 5
        if Shared:
            feature_selection = [0] * 6
        
        feature_selection[0] = 1
        for i in range(1, len(config) + 1):
            if config[i-1]:
                feature_selection[i] = 1
        
        #if feature_selection[0] and feature_selection[1]:
        #    feature_selection[4] = 1
        #if feature_selection[2] and feature_selection[3]:
        #    feature_selection[5] = 1

        return feature_selection

if __name__ == '__main__':
    #Set Path with the folder location of the results
    path = Path("/scratch/messerig/EvaluationScripts/ResultBlackbox/GroundTruth/results")
        
    for results_path in path.iterdir():
        xz_config_to_time = parse_zip_files(results_path)
        regression_model = multible_linear_reagression(xz_config_to_time, results_path)
    
    #print(parse_zip_files(path))
    #xz_config_to_time = parse_zip_files(path)
    #regression_model = multible_linear_reagression(xz_config_to_time)