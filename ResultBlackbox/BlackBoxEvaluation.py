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
import copy


def xz_string_to_tuple(file_name:str):
    #takes string containing xz filename and transform it to a 3 elements tuple (compression, extreme, threads)
    result = list()
    file_name = file_name.split("/")[-1]
    for chars in file_name:
        if chars.isnumeric():
            result.append(int(chars))
    return tuple(result)

def parse_zip_files(path:Path):
    #Used to meassure the mean of the time spend of for each zip file containing multiple time reports
    #path: Folder containing mutliple zips
    
    #Maps config (compression, extreme, threads) to time it took to run xz
    feature_config_to_time = dict()
    for zip in path.iterdir():
        xz_features = xz_string_to_tuple(str(zip))
        xz_result = TimeReportAggregate(zip)
        feature_config_to_time[xz_features] = py.mean(xz_result.measurements_wall_clock_time)
    return feature_config_to_time

class multible_linear_reagression:
    
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
        
     
    def generate_table(self, row_names:List, column_names:List, opt:List=[]):
        """
        This function generates the tables for the xz results

        Args:
            row_names (List): row names
            column_names (List): column names
            opt (List, optional): If Extreme is on. Defaults to [].
        """
        
        data = [[0] * len(column_names) for _ in range(len(row_names))]
        
        for y,row in enumerate(row_names):
            for x,column in enumerate(column_names):
                if (row + column) == []:
                    if opt:
                        position = self.filtered_config.index(opt)
                        data[y][x] = self.linear_model.coef_[position]
                    else:    
                        data[y][x] = self.linear_model.intercept_
                    continue
                if opt:
                    if (row == []):
                        position = self.filtered_config.index(row + ["compression_level_0"] + opt + column )
                        data[y][x] = self.linear_model.coef_[position]
                        continue
                position = self.filtered_config.index(row + opt + column)
                data[y][x] = self.linear_model.coef_[position]
        
        column_names_cpy = column_names[:]
        row_names_cpy = row_names[:]
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = round(data[i][j],ndigits= 3)
            
        column_names_cpy[0] = ["Base"]
        row_names_cpy[0] = ["Base"]
        column_names_cpy = [x[0] for x in column_names_cpy]
        row_names_cpy = [x[0] for x in row_names_cpy]
    
        result_table = pd.DataFrame(data=data, columns=column_names_cpy, index=row_names_cpy)
        print(result_table.to_latex())
            
        data_similarity = copy.deepcopy(data)
        #Hard coded WB data since we have only 5 values i did not implement a functionality to read that from a file
        if not opt:
            wb_data = [36.231, 0.095, 0.073, 0.072, 0.074]
            for i in range(5):
                data_similarity[0][i] = data_similarity[0][i] - wb_data[i]
        
        
        for i in range(len(data_similarity)):
            for j in range(len(data_similarity[i])):
                data_similarity[i][j] = abs(data_similarity[i][j]) / self.overall_runtime
                self.similarity_score += data_similarity[i][j]
        print("\n")

        result_table_similarity = pd.DataFrame(data=data_similarity, columns=column_names_cpy, index=row_names_cpy)
        print(result_table_similarity.to_latex())

    def generate_model(self):
        """generates mapping from configuration id to the features that are active for that id

        """
        feature_name = [[] for _ in range(105)]
        self.name_inverse = dict()
        for compression_level in range(10):
            feature_name[compression_level] = [f"compression_level_{compression_level}"]
            self.name_inverse[f"compression_level_{compression_level}"] = compression_level
        feature_name[10] = ["extreme"]
        self.name_inverse["extreme"] = 1
        for threads in range(0,4):
            feature_name[11 + threads] = [f"threads_{1 << threads}"]
            self.name_inverse[f"threads_{1 << threads}"] = 1 << threads
        for i in range(15,25):
            feature_name[i] = feature_name[i-15] + ["extreme"]
            
        offset = 25
        
        for compression_level in range(10):
            for extreme in range(2):
                for threads in range(0,4):
                    if extreme:
                        feature_name[offset + (compression_level * 8) + 4 + threads] = feature_name[compression_level] \
                                                                                        + ["extreme"] + [f"threads_{1 << threads}"]
                    else:
                        feature_name[offset + (compression_level * 8) + threads] = feature_name[compression_level] \
                                                                                        + [f"threads_{1 << threads}"]
        
        return feature_name
    
    def __init__(self, feature_config_to_time:dict):
        """Builds the performance influence model for XZ black-box using linear regression and prints a latex file
        Also calculates the similarity scores

        Args:
            feature_config_to_time (dict): _description_
        """
        
        
        self.feature_name = self.generate_model()
        
        X = list()
        y = list()
        
        for config_time_pair in feature_config_to_time.items():
            
            config = config_time_pair[0]
            time_spent = config_time_pair[1]
            
            y.append(time_spent)
            X.append(self.config_to_matrix(config))  
        
        names = [self.feature_name[i][0] for i in range(15)]
        
        x_test = list()
        for elements in X:
            x_test.append([str(i) for i in elements])
        
        self.configurations = pd.DataFrame(x_test, columns=names, dtype=str)
        self.configurations['performance'] = pd.Series(y)
        
        self.filtered_config = self.apply_iterative_vif(self.feature_name, nfp="performance")
        
        
        #filtered_config.remove(['compression_level_0'])    
        regression_data = [[0] * len(self.filtered_config) for _ in range(len(y))]
        dependent_data_sorted = list()
        
        for i in range(len(y)):
            config = self.filtered_config[i]
            config_tuple = [0,0,0]
            regression_data[i][self.name_inverse["compression_level_0"]] = 1
            for feature in config:
                feature = [feature]
                regression_data[i][self.feature_name.index(feature)] = 1
                if "compression" in feature[0]:
                    config_tuple[0] = self.name_inverse[feature[0]]
                if "extreme" in feature[0]:
                    config_tuple[1] = self.name_inverse[feature[0]]
                if "threads" in feature[0]:
                    config_tuple[2] = self.name_inverse[feature[0]]
            regression_data[i][self.filtered_config.index(config)] = 1
            dependent_data_sorted.append(feature_config_to_time[tuple(config_tuple)])
                

           
        self.X = py.array(regression_data)
        self.y = py.array(dependent_data_sorted)
        self.overall_runtime = sum(self.y)
        self.linear_model = LinearRegression().fit(self.X, self.y)
        table_y = [[]]
        table_y = table_y + self.filtered_config[1:10]
        table_x = [[]]
        table_x = table_x + self.filtered_config[11:15]
        self.similarity_score = 0
        self.generate_table(table_y, table_x)
        self.generate_table(table_y, table_x, opt=["extreme"])    

        print(round(self.similarity_score / 100, ndigits=6))
        
        print(self.overall_runtime, "\n")
        print(self.similarity_score / 100 * self.overall_runtime)
        
    @property
    def score(self):
        return self.linear_model.score(self.X, self.y)
    
    @property
    def feature_list(self):
        return self.feature_name
    

    def config_to_matrix(self, config:tuple):
        #Receives a 3 element tuple and returns a list that corresponds to the feature
        
        #24 solo features 
        #8 feature interactions per compression level
        feature_selection = [0] * 15
        #feature_selection = [0] * 104
        
        compression_level = config[0]
        extreme = config[1]
        threads = config[2]
        
        
        #0 is base -> alwads on
        feature_selection[0] = 1
        #0-9 => compression
        feature_selection[compression_level] = 1
        
        
        if extreme:
            #10-19 => extreme
            feature_selection[10] = 1
            
        #11-14 => Threads
        if threads:
            threads = (int(py.log2(config[2])))
            feature_selection[11 + threads] = 1
            #feature_selection[20 + threads] = 1
                                                                  
        return feature_selection


if __name__ == '__main__':
    #Set Path with the folder location of the results
    path = Path("/scratch/messerig/EvaluationScripts/ResultBlackbox/results")
        
    #print(parse_zip_files(path))
    xz_config_to_time = parse_zip_files(path)
    regression_model = multible_linear_reagression(xz_config_to_time)

    coefficients = list(regression_model.linear_model.coef_)