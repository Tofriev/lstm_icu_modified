import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.impute import KNNImputer
from utils import set_seed

set_seed(42)

class Preprocessor:
    def __init__(self, data, variables, parameters):
        self.data = data
        self.variables = variables
        self.parameters = parameters
        self.aggregation_freq = self.parameters.get('aggregation_frequency', 'H')
        self.imputation = self.parameters['imputation']
        self.sequence_dict = {}

    def process(self):
        if self.parameters['dataset_type'] == 'tudd_tudd':
            print('Processing TUDD data...')
            self.process_tudd()
        elif self.parameters['dataset_type'] == 'mimic_mimic':
            print('Processing MIMIC data...')
            self.process_mimic()
        elif self.parameters['dataset_type'] == 'mimic_tudd' or self.parameters['dataset_type'] == 'tudd_mimic':
            print('Processing MIMIC data...')
            self.process_mimic()
            print('Processing TUDD data...')
            self.process_tudd()

        return self.sequence_dict
            
    def process_mimic(self):
        self.variable_conversion_and_aggregation()
        self.create_time_grid()
        self.merge_on_time_grid()
        self.make_feature_lists() 

        self.impute()
        if self.parameters.get('sampling', False) != False:
            self.sample()
        self.scale()
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
        if 'height' in self.merged_df.columns:
            NUMERICAL_FEATURES.append('height')
        CAT_FEATURES = []
        if 'gender' in self.merged_df.columns:
            CAT_FEATURES.append('gender')
        self.SEQUENCE_FEATURES = SEQUENCE_FEATURES
        self.NUMERICAL_FEATURES = NUMERICAL_FEATURES
        self.CAT_FEATURES = CAT_FEATURES
        self.ALL_FEATURES = NUMERICAL_FEATURES + CAT_FEATURES


    def impute(self):
        X = self.merged_df
        if self.imputation['method'] == 'mean':
            self.impute_with_mean(X)
        elif self.imputation['method'] == 'rolling_mean':
            self.impute_with_rolling_mean(X)
        elif self.imputation['method'] == 'ffill_bfill':
            self.impute_with_ffill_bfill(X)
        elif self.imputation['method'] == 'knn':
            self.impute_with_knn(X)
        self.merged_df = X
        print(self.merged_df.head())
    
    def sample(self):
        method = self.parameters['sampling']['method']
        if method == 'undersampling':
            X = self.merged_df
            y = X[['stay_id', self.parameters['target']]].drop_duplicates()
            undersampler = RandomUnderSampler(random_state=42, sampling_strategy=self.parameters['sampling']['sampling_strategy']) 
            y_undersampled_stayids, _ = undersampler.fit_resample(y[['stay_id']], y[self.parameters['target']])
            stay_ids = y_undersampled_stayids['stay_id']
            self.merged_df = X[X['stay_id'].isin(stay_ids)]

    def scale(self):
        if self.parameters['scaling'] == 'standard':
            X = self.merged_df
            scaler = StandardScaler()
            X[self.NUMERICAL_FEATURES] = scaler.fit_transform(X[self.NUMERICAL_FEATURES])
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
        X = self.merged_df
        sequences = []
        for stay_id, group in X.groupby('stay_id'):
            group = group.sort_values('charttime')
            features = group[self.ALL_FEATURES].values
            label = group[self.parameters['target']].iloc[0]
            sequences.append((features, label))
        self.sequences = sequences

    def split_train_test_sequences(self):
        labels = [seq[1] for seq in self.sequences]
        self.sequence_dict['mimic'] = {}
        self.sequence_dict['mimic']['train'], self.sequence_dict['mimic']['test'] = train_test_split(self.sequences, test_size=0.2, stratify=labels, random_state=42)

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

    def impute_with_ffill_bfill(self, X):
        self.find_variables_with_only_nan(X)
        X.sort_values(['stay_id', 'charttime'], inplace=True)
        global_means = X[self.NUMERICAL_FEATURES].mean()
        print(f'global means {global_means}')
        for num_feature in self.SEQUENCE_FEATURES:
            X[num_feature] = X.groupby('stay_id')[num_feature].ffill()
            X[num_feature] = X.groupby('stay_id')[num_feature].bfill()

        # mean imputation for features without any vals 
        X[self.NUMERICAL_FEATURES] = X[self.NUMERICAL_FEATURES].fillna(global_means)

        for cat_feature in self.CAT_FEATURES:
            X[cat_feature].fillna(X[cat_feature].mode()[0], inplace=True)


    def impute_with_knn(self, X):
        knn_imputer = KNNImputer(n_neighbors=5) 
        X[self.NUMERICAL_FEATURES] = knn_imputer.fit_transform(X[self.NUMERICAL_FEATURES])

    def find_variables_with_only_nan(self, X):
        """
        Prints the number of stay_id's that have only NaN values for each variable (feature).
        """
        variables_with_nan_count = {}

        for feature in X.columns:
            stay_ids_with_nan = X.groupby('stay_id')[feature].apply(lambda group: group.isnull().all()).sum()

            if stay_ids_with_nan > 0:
                variables_with_nan_count[feature] = stay_ids_with_nan

        if variables_with_nan_count:
            print("Variables with the number of stay_ids that have only NaN values:")
            for variable, count in variables_with_nan_count.items():
                print(f"{variable}: {count} stay_id(s) with only NaN values")
        else:
            print("No variables have only NaN values across any stay_id.")

        return variables_with_nan_count
    

    def process_tudd(self):
        measurements = self.data['tudd']['measurements']
        mortality_info = self.data['tudd']['mortality_info']

        self.SEQUENCE_FEATURES = [
        'hr_value', 'mbp_value', 'total_gcs', 'glc_value', 'creatinine_value', 
        'potassium_value', 'wbc_value', 'platelets_value', 'inr_value', 
        'anion_gap_value', 'lactate_value', 'temperature_value', 'weight_value'
        ]
        self.NUMERICAL_FEATURES = self.SEQUENCE_FEATURES + ['age', 'height_value']
        self.CAT_FEATURES = ['gender']
        self.ALL_FEATURES = self.NUMERICAL_FEATURES + self.CAT_FEATURES

        measurements['measurement_offset'] = pd.to_numeric(measurements['measurement_offset'], errors='coerce')

        measurements = pd.merge(
            measurements,
            mortality_info[['caseid','stay_duration', 'age', 'gender', 'bodyheight', 'bodyweight', 'exitus']],
            on='caseid',
            how='left'
        )

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
        measurements_agg = measurements.groupby(['caseid', 'measurement_time_from_admission', 'treatmentname'])['value'].mean().reset_index()

        # pivot
        measurements_pivot = measurements_agg.pivot_table(
            index=['caseid', 'measurement_time_from_admission'],
            columns='treatmentname',
            values='value'
        ).reset_index()

        # make time grid
        def create_time_grid(mortality_info):
            df_list = []
            for _, row in mortality_info.iterrows():
                caseid = row['caseid']
                time_range = np.arange(0, 25)  # hour 0 to 24 inclusive
                time_df = pd.DataFrame({'caseid': caseid, 'measurement_time_from_admission': time_range})
                df_list.append(time_df)
            return pd.concat(df_list, ignore_index=True)

        time_grid = create_time_grid(mortality_info)

        # merg on time grid
        merged_df = pd.merge(time_grid, measurements_pivot, on=['caseid', 'measurement_time_from_admission'], how='left')
        merged_df = pd.merge(
            merged_df,
            mortality_info[['caseid', 'age', 'gender', 'bodyheight', 'bodyweight', 'exitus']],
            on='caseid',
            how='left'
        )

        # rename
        treatmentnames_mapping = {
            'HF': 'hr_value', 'AGAP': 'anion_gap_value', 'GLUC': 'glc_value',
            'CREA': 'creatinine_value', 'K': 'potassium_value', 'LEU': 'wbc_value',
            'THR': 'platelets_value', 'Q': 'inr_value', 'LAC': 'lactate_value',
            'T': 'temperature_value', 'GCS': 'total_gcs', 'MAP': 'mbp_value',
            'bodyweight': 'weight_value', 'bodyheight': 'height_value'
        }
        merged_df.rename(columns=treatmentnames_mapping, inplace=True)

        # bounds 
        bounds = {
            'age': (18, 90), 'weight_value': (20, 500), 'height_value': (20, 260),
            'temperature_value': (20, 45), 'hr_value': (10, 300), 'glc_value': (5, 2000),
            'mbp_value': (20, 400), 'potassium_value': (2.5, 7), 'wbc_value': (1, 200),
            'platelets_value': (10, 1000), 'inr_value': (0.2, 6), 'anion_gap_value': (1, 25),
            'lactate_value': (0.1, 200), 'creatinine_value': (0.1, 20)
        }

        merged_df = merged_df[merged_df['age'] >= 18]
        merged_df['age'] = merged_df['age'].apply(lambda x: min(x, 90))

        for feature, (lower, upper) in bounds.items():
            if feature in merged_df.columns:
                merged_df.loc[merged_df[feature] < lower, feature] = np.nan
                merged_df.loc[merged_df[feature] > upper, feature] = np.nan

        # imputation
        merged_df.sort_values(['caseid', 'measurement_time_from_admission'], inplace=True)
        merged_df = merged_df.groupby('caseid').apply(lambda group: group.ffill().bfill()).reset_index(drop=True)

        global_means = merged_df[self.NUMERICAL_FEATURES].mean()
        merged_df[self.NUMERICAL_FEATURES] = merged_df[self.NUMERICAL_FEATURES].fillna(global_means)

        # categorical features 
        merged_df['gender'] = merged_df['gender'].map({'m': 0, 'w': 1})
        merged_df['gender'].fillna(merged_df['gender'].mode()[0], inplace=True)
        merged_df['exitus'].fillna(0, inplace=True)

        scaler = StandardScaler()
        merged_df[self.NUMERICAL_FEATURES] = scaler.fit_transform(merged_df[self.NUMERICAL_FEATURES])

        column_order = [
            'caseid', 'measurement_time_from_admission', 'mbp_value', 'total_gcs', 'glc_value',
            'creatinine_value', 'hr_value', 'potassium_value', 'wbc_value', 'platelets_value',
            'inr_value', 'anion_gap_value', 'lactate_value', 'temperature_value', 'weight_value',
            'age', 'gender', 'height_value', 'exitus'
        ]
        sorted_merged_df = merged_df[column_order]

        sequences = []
        for caseid, group in sorted_merged_df.groupby('caseid'):
            if len(group) == 25:
                features = group[self.ALL_FEATURES].values
                label = group['exitus'].iloc[0]
                sequences.append((features, label))

        labels = [seq[1] for seq in sequences]
        self.sequence_dict['tudd'] = {}
        self.sequence_dict['tudd']['train'], self.sequence_dict['tudd']['test'] = train_test_split(sequences, test_size=0.2, stratify=labels, random_state=42)
