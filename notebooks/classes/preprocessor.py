import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer
from utils import set_seed
import matplotlib.pyplot as plt
import seaborn as sns
import random

set_seed(42)

class Preprocessor:
    def __init__(self, data, variables, parameters, mimic_scaler=None, ALL_FEATURES_MIMIC=None):
        self.data = data
        self.variables = variables
        self.parameters = parameters
        self.mimic_scaler = mimic_scaler
        self.ALL_FEATURES_MIMIC = ALL_FEATURES_MIMIC
        self.aggregation_freq = self.parameters.get('aggregation_frequency', 'H')
        self.imputation = self.parameters['imputation']
        if self.imputation['method'] == 'knn':
            self.imputation['n_neighbors'] = self.parameters.get('n_neighbors', 5)
        self.sequence_dict = {}
        self.plot_dict = {}
        self.compare_distributions = self.parameters.get('compare_distributions', False)
        self.shuffle = self.parameters.get('shuffle', False)

    def process(self):
        sequences_dict = {}

        if 'mimic' in self.data:
            print('Processing MIMIC data...')
            self.process_mimic()
            sequences_dict['mimic'] = self.sequence_dict['mimic']
            self.feature_index_mapping = {index: feature for index, feature in enumerate(self.ALL_FEATURES)}

        if 'tudd' in self.data:
            print('Processing TUDD data...')
            self.process_tudd()
            sequences_dict['tudd'] = self.sequence_dict['tudd']
            self.feature_index_mapping = {index: feature for index, feature in enumerate(self.ALL_FEATURES)}

        if self.parameters.get('fractional_steps') and self.parameters['dataset_type'] == 'mimic_tudd_fract':
            self.generate_fractions()



        if self.compare_distributions:
            if 'tudd' in self.parameters['dataset_type'] and 'mimic' in self.parameters['dataset_type']:
                mimic_df = self.plot_dict['mimic']
                tudd_df = self.plot_dict['tudd']
                mimic_imputed_df = self.plot_dict['mimic_imputed']
                tudd_imputed_df = self.plot_dict['tudd_imputed']
                print('Distributions before imputation:')
                self.plot_density(mimic_df, tudd_df, self.MIMIC_NUMERICAL_FEATURES)
                print('Distributions after imputation:')
                self.plot_density(mimic_imputed_df, tudd_imputed_df, self.MIMIC_NUMERICAL_FEATURES)
            
        return self.sequence_dict, self.feature_index_mapping
            
    def process_mimic(self):
        self.variable_conversion_and_aggregation()
        self.create_time_grid()
        merged = self.merge_on_time_grid()
        self.make_feature_lists() 
        
        imputed_df = self.impute(merged)
        if self.compare_distributions:
            self.plot_dict['mimic_imputed'] = imputed_df
        self.merged_df = imputed_df
        if self.parameters.get('sampling', False) != False:
            self.sample()
        self.scale_normalize()
        self.check()

        self.create_sequences()
        self.split_train_test_sequences()

    def variable_conversion_and_aggregation(self):
        """
        converts vars in static and aggregate data on specified frequency
        """
        
        for variable, attr in self.variables.items():
            if variable == 'static_data':
                static_df = self.data['mimic']['static_data']
                static_df['intime'] = pd.to_datetime(static_df['intime']).dt.floor(self.aggregation_freq)
                static_df['first_day_end'] = pd.to_datetime(static_df['first_day_end']).dt.floor(self.aggregation_freq)
                static_df['gender'] = static_df['gender'].map({'M': 0, 'F': 1})
                self.data['mimic']['static_data'] = static_df
            elif attr['sequence']:
                df = self.data['mimic'][variable]
                df['charttime'] = pd.to_datetime(df['charttime']).dt.floor(self.aggregation_freq)
                measurement_cols = df.columns.difference(['stay_id', 'charttime'])
                df_agg = df.groupby(['stay_id', 'charttime'], as_index=False)[measurement_cols.tolist()].mean()
                self.data['mimic'][variable] = df_agg

    def create_time_grid(self):
        static_df = self.data['mimic']['static_data']
        df_list = []
        for index, row in static_df.iterrows():
            stay_id = row['stay_id']
            start_time = row['intime']
            end_time = row['first_day_end']
            time_range = pd.date_range(start=start_time, end=end_time, freq=self.aggregation_freq)
            time_df = pd.DataFrame({'stay_id': stay_id, 'charttime': time_range})
            df_list.append(time_df)
        self.time_grid = pd.concat(df_list, ignore_index=True)

    def merge_on_time_grid(self):
        merged_df = self.time_grid.copy()
        for variable, attrs in self.variables.items():
            if variable == 'static_data':
                continue
            if attrs['sequence']:
                df = self.data['mimic'][variable]
                merged_df = pd.merge(merged_df, df, on=['stay_id', 'charttime'], how='left')
        static_columns = [col for col in self.data['mimic']['static_data'].columns if col not in ['intime', 'first_day_end']]
        merged_df = pd.merge(merged_df, self.data['mimic']['static_data'][static_columns], on='stay_id', how='left')
        self.merged_df = merged_df
        if self.compare_distributions:
            self.plot_dict['mimic'] = merged_df
        return merged_df
    
    def make_feature_lists(self):
        SEQUENCE_FEATURES = []
        for variable, attrs in self.variables.items():
            if variable == 'static_data':
                continue
            if attrs['sequence'] and attrs['type'] == 'numerical':
                SEQUENCE_FEATURES.append(f"{variable}_value")
        NUMERICAL_FEATURES = SEQUENCE_FEATURES + []
        if 'age' in self.merged_df.columns:
            NUMERICAL_FEATURES.append('age')
        # if 'height' in self.merged_df.columns:
        #     NUMERICAL_FEATURES.append('height_value')
        CAT_FEATURES = []
        if 'gender' in self.merged_df.columns:
            CAT_FEATURES.append('gender')
        self.SEQUENCE_FEATURES = SEQUENCE_FEATURES
        self.NUMERICAL_FEATURES = NUMERICAL_FEATURES
        self.MIMIC_NUMERICAL_FEATURES = NUMERICAL_FEATURES
        self.CAT_FEATURES = CAT_FEATURES
        self.ALL_FEATURES = NUMERICAL_FEATURES + CAT_FEATURES


    def impute(self, input_df):
        renamed = False
        if 'measurement_time_from_admission' in input_df.columns:
            input_df.rename(columns={'measurement_time_from_admission': 'charttime'}, inplace=True)
            renamed = True
        if self.imputation['method'] == 'mean':
            imputed_df = self.impute_with_mean(input_df)
        elif self.imputation['method'] == 'rolling_mean':
            imputed_df = self.impute_with_rolling_mean(input_df)
        elif self.imputation['method'] == 'ffill_bfill':
            imputed_df = self.impute_with_ffill_bfill(input_df)
        elif self.imputation['method'] == 'knn':
            imputed_df = self.impute_with_knn(input_df)
        
        # impute remaining missing values with global mean
        imputed_df[self.NUMERICAL_FEATURES] = imputed_df[self.NUMERICAL_FEATURES].fillna(imputed_df[self.NUMERICAL_FEATURES].mean())
        if renamed:
            imputed_df.rename(columns={'charttime': 'measurement_time_from_admission'}, inplace=True)
        return imputed_df
    
    def sample(self):
        method = self.parameters['sampling']['method']
        if method == 'undersampling':
            X = self.merged_df
            y = X[['stay_id', self.parameters['target']]].drop_duplicates()
            undersampler = RandomUnderSampler(random_state=42, sampling_strategy=self.parameters['sampling']['sampling_strategy']) 
            y_undersampled_stayids, _ = undersampler.fit_resample(y[['stay_id']], y[self.parameters['target']])
            stay_ids = y_undersampled_stayids['stay_id']
            self.merged_df = X[X['stay_id'].isin(stay_ids)]

    def scale_normalize(self):
        if self.parameters['scaling'] == 'standard':
            X = self.merged_df
            mimic_scaler = StandardScaler()
            X[self.NUMERICAL_FEATURES] = mimic_scaler.fit_transform(X[self.NUMERICAL_FEATURES])
            self.mimic_scaler = mimic_scaler
            self.merged_df = X
        elif self.parameters['scaling'] == 'MinMax':
            X = self.merged_df
            mimic_scaler = MinMaxScaler(feature_range=(self.parameters['scaling_range'][0], self.parameters['scaling_range'][1]))
            X[self.NUMERICAL_FEATURES] = mimic_scaler.fit_transform(X[self.NUMERICAL_FEATURES])
            self.MIMIC_NUMERICAL_FEATURES = self.NUMERICAL_FEATURES
            self.mimic_scaler = mimic_scaler
            self.merged_df = X

    def check(self):
        X = self.merged_df
        stay_ids_with_missing_rows = []
        for stay_id, group in X.groupby('stay_id'):
            if len(group) != 25:        
                stay_ids_with_missing_rows.append(stay_id)
        if stay_ids_with_missing_rows:
            print("Stay IDs with != 25 rows:", stay_ids_with_missing_rows)
        else:
            print("All stay IDs have 25 rows.")

    def create_sequences(self):
        print(f"Number of unique stay_ids: {self.merged_df['stay_id'].nunique()}")
        print(f'MIMIC features: {self.merged_df.columns}')  
        print(f'n_MIMIC: {len(self.merged_df.columns)}')
        print(self.merged_df.describe())
        X = self.merged_df
        sequences = []
        print(f'length MIMIC ALL FEATURES: {len(self.ALL_FEATURES)}')
        print(self.ALL_FEATURES)
        for stay_id, group in X.groupby('stay_id'):
            group = group.sort_values('charttime')
            features = group[self.ALL_FEATURES].values
            label = group[self.parameters['target']].iloc[0]
            sequences.append((features, label))
        self.ALL_FEATURES_MIMIC = self.ALL_FEATURES
        self.sequences = sequences
        self.feature_index_mapping = {index: feature for index, feature in enumerate(self.ALL_FEATURES)}

    def split_train_test_sequences(self):
        labels = [seq[1] for seq in self.sequences]
        self.sequence_dict['mimic'] = {}
        self.sequence_dict['mimic']['train'], self.sequence_dict['mimic']['test'] = train_test_split(self.sequences, test_size=0.2, stratify=labels, random_state=42)
        print(self.sequence_dict['mimic']['test'][0][0].shape)

    def impute_with_mean(self, X):
        global_means = X[self.NUMERICAL_FEATURES].mean()
        X[self.NUMERICAL_FEATURES] = X[self.NUMERICAL_FEATURES].fillna(global_means)
        for cat_feature in self.CAT_FEATURES:
            X[cat_feature].fillna(X[cat_feature].mode()[0], inplace=True)

    

    def impute_with_rolling_mean(self, X):
        X = X.sort_values(['stay_id', 'charttime'])
        X[self.NUMERICAL_FEATURES] = X.groupby('stay_id')[self.NUMERICAL_FEATURES].apply(
            lambda group: group.fillna(group.rolling(window=3, min_periods=1).mean())
        )

    def impute_with_ffill_bfill(self, df):
        df.sort_values(['stay_id', 'charttime'], inplace=True)
        for num_feature in self.SEQUENCE_FEATURES:
            df[num_feature] = df.groupby('stay_id')[num_feature].ffill()
            df[num_feature] = df.groupby('stay_id')[num_feature].bfill()



        
        
    #     stay_ids_to_drop = X.groupby('stay_id')[self.NUMERICAL_FEATURES].apply(
    #         lambda group: group.isnull().all(axis=0).any()
    #     )
    #     stay_ids_to_drop = stay_ids_to_drop[stay_ids_to_drop].index

    #     print(f"Number of MIMIC observations before dropping: {len(X)}")

    #     #DROP
    #    # X = X[~X['stay_id'].isin(stay_ids_to_drop)]
    #     print(f"Number of MIMIC observations after dropping: {len(X)}")
    #     # # Use KNN imputation for features without any values
    #     # features_to_impute = self.NUMERICAL_FEATURES
    #     # print('starting knn imputation as part of ffill_bfill')
    #     # knn_imputer = KNNImputer(n_neighbors=4)
    #     # X[features_to_impute] = knn_imputer.fit_transform(X[features_to_impute])
    #     # print('knn imputation done')
    #     # for cat_feature in self.CAT_FEATURES:
    #     #     X[cat_feature].fillna(X[cat_feature].mode()[0], inplace=True)
        return df
 
    def impute_with_knn(self, X):
        print("Starting KNN imputation...")
        
        features_to_impute = self.NUMERICAL_FEATURES # + self.CAT_FEATURES
        
        knn_imputer = KNNImputer(n_neighbors=self.imputation['n_neighbors'])

        X[features_to_impute] = knn_imputer.fit_transform(X[features_to_impute])
        
        for cat_feature in self.CAT_FEATURES:
            X[cat_feature].fillna(X[cat_feature].mode()[0], inplace=True)
        print("KNN imputation done.")
        return X
        

    def find_variables_with_only_nan(self, X):
        """
        Prints the number of stay_id's that have only NaN values for each variable (feature).
        """
        df = X.copy()
        variables_with_nan_count = {}
       
        for feature in df.columns:
            stay_ids_with_nan = df.groupby('stay_id')[feature].apply(lambda group: group.isnull().all()).sum()

            if stay_ids_with_nan > 0:
                variables_with_nan_count[feature] = stay_ids_with_nan
        if variables_with_nan_count:
            print("Variables with the number of stay_ids that have only NaN values:")
            for variable, count in variables_with_nan_count.items():
                print(f"{variable}: {count} stay_id(s) with only NaN values")
        else:
            print("No variables have only NaN values across any stay_id.")

        #return variables_with_nan_count
    
    def generate_fractions(self):
        print('Generating fractional datasets...')
        mimic_train = self.sequence_dict['mimic']['train']
        mimic_test = self.sequence_dict['mimic']['test']
        tudd_train = self.sequence_dict['tudd']['train']
        tudd_test = self.sequence_dict['tudd']['test']

        n_tudd_train = len(tudd_train) - 2000
        step_size = self.parameters['fractional_steps']
        fractional_datasets = {}
        n_sampled_tudd_train = 0
        
        while n_sampled_tudd_train+step_size < n_tudd_train:
            n_sampled_tudd_train += step_size

            # getg next tudd batch
            tudd_samples = tudd_train[:n_sampled_tudd_train]
            combined_train_set = mimic_train + tudd_samples
            if self.shuffle == True:
                random.shuffle(combined_train_set)

            fractional_datasets[n_sampled_tudd_train] = combined_train_set

            print(f'fraction {n_sampled_tudd_train} added')

        self.sequence_dict['fractional_mimic_tudd'] = fractional_datasets

    def plot_density(self, mimic_df, tudd_df, features):

        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(mimic_df[feature].dropna(), label='MIMIC', fill=True, alpha=0.5)
            sns.kdeplot(tudd_df[feature].dropna(), label='TUDD', fill=True, alpha=0.5)
            plt.title(f'Density Plot for {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            print(f"Mean {feature} MIMIC: {mimic_df[feature].mean()}")
            print(f"Mean {feature} TUDD: {tudd_df[feature].mean()}")
      
    def process_tudd(self):
        measurements = self.data['tudd']['measurements']
        mortality_info = self.data['tudd']['mortality_info']

        self.SEQUENCE_FEATURES = [
        'hr_value', 'mbp_value', 'gcs_total_value', 'glc_value', 'creatinine_value', 
        'potassium_value', 'wbc_value', 'platelets_value', 'inr_value', 
        'anion_gap_value', 'lactate_value', 'temperature_value', 'weight_value'
        ]
        self.NUMERICAL_FEATURES = self.SEQUENCE_FEATURES + ['age']#, 'height_value']
        self.CAT_FEATURES = ['gender']
        self.ALL_FEATURES = self.NUMERICAL_FEATURES + self.CAT_FEATURES

        measurements['measurement_offset'] = pd.to_numeric(measurements['measurement_offset'], errors='coerce')

        measurements = pd.merge(
            measurements,
            mortality_info[['caseid','stay_duration', 'age', 'gender',  'bodyweight', 'exitus']], #'bodyheight',
            on='caseid',
            how='left'
        )
        measurements.rename(columns={'caseid': 'stay_id'}, inplace=True)
        mortality_info.rename(columns={'caseid': 'stay_id'}, inplace=True)
        measurements['stay_duration_hours'] = measurements['stay_duration'] * 24
        measurements['measurement_time_from_admission'] = measurements['stay_duration_hours'] + measurements['measurement_offset']

        # clean negative vals
        measurements = measurements[measurements['measurement_time_from_admission'] > -1]
        measurements.loc[measurements['measurement_time_from_admission'] <= -1, 'measurement_time_from_admission'] = 0

        # filter first 24 h
        measurements = measurements[(measurements['measurement_time_from_admission'] >= 0) &
                                    (measurements['measurement_time_from_admission'] <= 24)]
        measurements['measurement_time_from_admission'] = np.floor(measurements['measurement_time_from_admission'])

        # aggregate
        measurements['value'] = pd.to_numeric(measurements['value'], errors='coerce')
        measurements_agg = measurements.groupby(['stay_id', 'measurement_time_from_admission', 'treatmentname'])['value'].mean().reset_index()

        # pivot
        measurements_pivot = measurements_agg.pivot_table(
            index=['stay_id', 'measurement_time_from_admission'],
            columns='treatmentname',
            values='value'
        ).reset_index()

    
        # make time grid
        def create_time_grid(mortality_info):
            df_list = []
            for _, row in mortality_info.iterrows():
                stay_id = row['stay_id']
                time_range = np.arange(0, 25)  # hour 0 to 24 inclusive
                time_df = pd.DataFrame({'stay_id': stay_id, 'measurement_time_from_admission': time_range})
                df_list.append(time_df)
            return pd.concat(df_list, ignore_index=True)

        time_grid = create_time_grid(mortality_info)

        # merg on time grid
        merged_df = pd.merge(time_grid, measurements_pivot, on=['stay_id', 'measurement_time_from_admission'], how='left')
        merged_df = pd.merge(
            merged_df,
            mortality_info[['stay_id', 'age', 'gender', 'bodyweight', 'exitus']], # 'bodyheight', 
            on='stay_id',
            how='left'
        )

        # rename
        treatmentnames_mapping = {
            'HF': 'hr_value', 'AGAP': 'anion_gap_value', 'GLUC': 'glc_value',
            'CREA': 'creatinine_value', 'K': 'potassium_value', 'LEU': 'wbc_value',
            'THR': 'platelets_value', 'Q': 'inr_value', 'LAC': 'lactate_value',
            'T': 'temperature_value', 'GCS': 'gcs_total_value', 'MAP': 'mbp_value',
            'bodyweight': 'weight_value'#, 'bodyheight': 'height_value'
        }
        merged_df.rename(columns=treatmentnames_mapping, inplace=True)
        print(f'number of unique stay_ids before renaming and bounding: {merged_df["stay_id"].nunique()}')
        # bounds 
        bounds = {
            'age': (18, 90), 'weight_value': (20, 500), #'height_value': (20, 260),
            'temperature_value': (20, 45), 'hr_value': (10, 300), 'glc_value': (5, 2000),
            'mbp_value': (20, 400), 'potassium_value': (2.5, 7), 'wbc_value': (1, 200),
            'platelets_value': (10, 1000), 'inr_value': (0.2, 6), 'anion_gap_value': (1, 25),
            'lactate_value': (0.1, 200), 'creatinine_value': (0.1, 20)
        }
        print(f'number of unique stay_ids after bounding: {merged_df["stay_id"].nunique()}')
        # if self.parameters['small_data']:
        #     fraction = 0.1
        #     patient_sample = merged_df[['stay_id', 'exitus']].drop_duplicates().groupby('exitus', group_keys=False).apply(
        #         lambda x: x.sample(frac=fraction, random_state=42)
        #     )
        #     sampled_df = merged_df[merged_df['stay_id'].isin(patient_sample['stay_id'])]
        #     merged_df = sampled_df

        # mean before conversion
        print("Mean before conversion:")
        print(f"Glucose (mmol/L): {merged_df['glc_value'].mean()}")
        print(f"Creatinine (micro_mol/L): {merged_df['creatinine_value'].mean()}")
        print(f"INR (Quick): {merged_df['inr_value'].mean()}")
        print(f"Lactate (mmol/L): {merged_df['lactate_value'].mean()}")

        # convert units
        # glucose mmol/L to mg/dL
        merged_df['glc_value'] = merged_df['glc_value'] * 18.0182
        print(f"Glucose conversion done: {merged_df['glc_value'].mean()} mg/dL")

        # creatinine micro_mol/L to mg/dL
        merged_df['creatinine_value'] = merged_df['creatinine_value'] * 0.0113
        print(f"Creatinine conversion done: {merged_df['creatinine_value'].mean()} mg/dL")

        # convert quick to inr 
        merged_df['inr_value'] = merged_df['inr_value'] / 100
        print(f"INR conversion done: {merged_df['inr_value'].mean()}")

        # convert lactate mmol/L to mg/dL
        # merged_df['lactate_value'] = merged_df['lactate_value'] * 9.01
        # print(f"Lactate conversion done: {merged_df['lactate_value'].mean()} mg/dL")
        print(f'number of unique stay_ids before filtering: {merged_df["stay_id"].nunique()}')
        # filter
        merged_df = merged_df[merged_df['age'] >= 18]
        merged_df['age'] = merged_df['age'].apply(lambda x: min(x, 90))

        for feature, (lower, upper) in bounds.items():
            if feature in merged_df.columns:
                merged_df.loc[merged_df[feature] < lower, feature] = np.nan
                merged_df.loc[merged_df[feature] > upper, feature] = np.nan
        merged_df['gender'] = merged_df['gender'].map({'m': 0, 'w': 1})
        if self.compare_distributions:
            self.plot_dict['tudd'] = merged_df

        if self.parameters['golden_tudd']:
            target_proportion_before = merged_df['exitus'].mean()
            print(f"Proportion of target before dropping: {target_proportion_before}")

            missing_counts = merged_df.groupby('stay_id')[self.ALL_FEATURES].apply(lambda x: x.isnull().sum().sum()) # total missing across all vars
            missing_counts = missing_counts.sort_values() # sort from least to most
            top_1000_stay_ids = missing_counts.index[:1000]
            merged_df = merged_df[merged_df['stay_id'].isin(top_1000_stay_ids)] 

            target_proportion_after = merged_df['exitus'].mean()
            print(f"Proportion of target after dropping: {target_proportion_after}")

            
            if merged_df['stay_id'].nunique() != 1000:
                raise ValueError(f"Expected 1000 unique stay_ids, but got {merged_df['stay_id'].nunique()}.")
           
        
        print(f'number of unique stay_ids before iumputing: {merged_df["stay_id"].nunique()}')
        # imputation
        #merged_df.sort_values(['stay_id', 'measurement_time_from_admission'], inplace=True)
        merged_df = self.impute(merged_df)
        # if self.imputation['method'] == 'ffill_bfill':
        #     merged_df = merged_df.groupby('caseid').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)
            
        #     self.find_variables_with_only_nan(merged_df)
        

        #     stay_ids_to_drop = merged_df.groupby('caseid')[self.NUMERICAL_FEATURES].apply(
        #     lambda group: group.isnull().all(axis=0).any()
        # )
        #     stay_ids_to_drop = stay_ids_to_drop[stay_ids_to_drop].index

        #     print(f"Number of TUDD observations before dropping: {len(merged_df)}")
        #     #merged_df = merged_df[~merged_df['caseid'].isin(stay_ids_to_drop)]
        #     print(f"Number of TUDD observations left after dropping: {len(merged_df)}")

        # elif self.imputation['method'] == 'knn':
        #     merged_df = self.impute_with_knn(merged_df)


      
        if self.compare_distributions:
            self.plot_dict['tudd_imputed'] = merged_df.copy()
        # categorical features 
        merged_df['gender'].fillna(merged_df['gender'].mode()[0], inplace=True)
        merged_df['exitus'].fillna(0, inplace=True)

        if self.parameters['scaling'] == 'standard':
            #scaler = StandardScaler()
            merged_df[self.MIMIC_NUMERICAL_FEATURES] = self.mimic_scaler.transform(merged_df[self.MIMIC_NUMERICAL_FEATURES])
        elif self.parameters['scaling'] == 'MinMax':
            #scaler = MinMaxScaler(feature_range=(self.parameters['scaling_range'][0], self.parameters['scaling_range'][1]))
            merged_df[self.MIMIC_NUMERICAL_FEATURES] = self.mimic_scaler.transform(merged_df[self.MIMIC_NUMERICAL_FEATURES])

        # column_order = [
        #     'caseid', 'measurement_time_from_admission', 'mbp_value', 'gcs_total_value', 'glc_value',
        #     'creatinine_value', 'potassium_value', 'hr_value', 'wbc_value', 'platelets_value',
        #     'lactate_value','temperature_value','weight_value', 'inr_value', 'anion_gap_value', 'exitus',   
        #     'age', 'gender', 'height_value'
        # ]
        column_order = [
             'stay_id', 'measurement_time_from_admission', 'mbp_value', 'gcs_total_value', 'glc_value',
             'creatinine_value', 'potassium_value', 'hr_value', 'wbc_value', 'platelets_value',
             'temperature_value','weight_value', 'exitus',   
             'age', 'gender'
         ]
        sorted_merged_df = merged_df[column_order]
        
        # # drop 'anion_gap_value'
        # sorted_merged_df.drop(columns=['anion_gap_value'], inplace=True)
        # self.SEQUENCE_FEATURES.remove('anion_gap_value')
        # self.NUMERICAL_FEATURES.remove('anion_gap_value')
        # self.ALL_FEATURES.remove('anion_gap_value')

        # # drop 'inr_value'
        # sorted_merged_df.drop(columns=['inr_value'], inplace=True)
        # self.SEQUENCE_FEATURES.remove('inr_value')
        # self.NUMERICAL_FEATURES.remove('inr_value')
        # self.ALL_FEATURES.remove('inr_value')

        # # drop lactate_value
        # sorted_merged_df.drop(columns=['lactate_value'], inplace=True)
        # self.SEQUENCE_FEATURES.remove('lactate_value')
        # self.NUMERICAL_FEATURES.remove('lactate_value')
        # self.ALL_FEATURES.remove('lactate_value')

        print(f"Number of unique stay_ids: {sorted_merged_df['stay_id'].nunique()}")

        sequences = []
        for stay_id, group in sorted_merged_df.groupby('stay_id'):
            if len(group) == 25:
                features = group[self.ALL_FEATURES_MIMIC].values
                label = group['exitus'].iloc[0]
                sequences.append((features, label))

        self.sequence_dict['tudd'] = {}
        if self.parameters['golden_tudd']:
            self.sequence_dict['tudd']['test'] = sequences
            self.sequence_dict['tudd']['train'] = []
        else:
            labels = [seq[1] for seq in sequences]
            self.sequence_dict['tudd']['train'], self.sequence_dict['tudd']['test'] = train_test_split(sequences, test_size=0.2, stratify=labels, random_state=42)
