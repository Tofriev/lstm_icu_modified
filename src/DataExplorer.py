import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DataExplorer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset.data
        self.variables = dataset.variables
        self.pivoted_data = dataset.pivoted_data
        self.mortality_data = self.data.drop_duplicates(subset=["stay_id"]).set_index(
            "stay_id"
        )["mortality"]

    def nan_summary(self):
        summary = {}
        for var in self.variables:
            pivoted_var = self.pivoted_data[var]
            nan_counts = pivoted_var.isna().sum()
            nan_summary = nan_counts[nan_counts > 0]
            summary[var] = nan_summary
            print(f"NaN Summary for {var}:\n{nan_summary}\n")
        return summary

    def plot_nan_heatmap(self):
        for var in self.variables:
            plt.figure(figsize=(12, 6))
            sns.heatmap(self.pivoted_data[var].isna(), cbar=False, cmap="viridis")
            plt.title(f"NaN Heatmap for {var}")
            plt.xlabel("Time Step")
            plt.ylabel("Patient ID")
            plt.show()

    def plot_nan_distribution(self):
        for var in self.variables:
            mortality_0 = self.mortality_data == 0
            mortality_1 = self.mortality_data == 1

            nan_counts_per_time_step_0 = (
                self.pivoted_data[var][mortality_0].isna().sum(axis=0)
                / mortality_0.sum()
            )
            nan_counts_per_time_step_1 = (
                self.pivoted_data[var][mortality_1].isna().sum(axis=0)
                / mortality_1.sum()
            )

            plt.figure(figsize=(12, 6))
            sns.lineplot(
                x=nan_counts_per_time_step_0.index,
                y=nan_counts_per_time_step_0.values,
                marker="o",
                label="Mortality = 0",
            )
            sns.lineplot(
                x=nan_counts_per_time_step_1.index,
                y=nan_counts_per_time_step_1.values,
                marker="o",
                label="Mortality = 1",
            )

            plt.title(
                f"Distribution of NaN Values for {var} (Normalized by Group Size)"
            )
            plt.xlabel("Time Step")
            plt.ylabel("Proportion of NaNs")
            plt.grid(True)
            plt.legend()
            plt.show()

    def plot_nan_distribution_by_mortality(self):
        for var in self.variables:
            nan_counts = self.pivoted_data[var].isna().sum(axis=1)
            data_with_mortality = pd.DataFrame(
                {"nan_counts": nan_counts, "mortality": self.mortality_data}
            )

            plt.figure(figsize=(12, 6))
            sns.boxplot(x="mortality", y="nan_counts", data=data_with_mortality)
            plt.title(f"NaN Counts by Mortality for {var}")
            plt.xlabel("Mortality")
            plt.ylabel("Number of NaNs")
            plt.show()

            median_nan_counts = data_with_mortality.groupby("mortality")[
                "nan_counts"
            ].median()
            print(f"Median NaN Counts for {var} by Mortality:\n{median_nan_counts}\n")

    def statistical_summary(self):
        summaries = {}
        for var in self.variables:
            summary = self.pivoted_data[var].describe()
            summaries[var] = summary
            print(f"Statistical Summary for {var}:\n{summary}\n")
        return summaries

    def plot_time_series(self, num_patients=5):
        for var in self.variables:
            plt.figure(figsize=(12, 6))
            sample_patients = self.pivoted_data[var].sample(
                num_patients, random_state=1
            )
            for idx, patient_data in sample_patients.iterrows():
                plt.plot(patient_data.values, label=f"Patient {idx}")
            plt.title(f"Time Series Plot for {var} (Sample of {num_patients} Patients)")
            plt.xlabel("Time Step")
            plt.ylabel(var)
            plt.legend()
            plt.show()

    def plot_histograms(self):
        for var in self.variables:
            plt.figure(figsize=(12, 6))
            self.pivoted_data[var].stack().hist(bins=50)
            plt.title(f"Histogram of {var}")
            plt.xlabel(var)
            plt.ylabel("Frequency")
            plt.show()

    def measurement_frequency_analysis(self):
        freq_summary = {}
        for var in self.variables:
            measurement_counts = self.pivoted_data[var].notna().sum(axis=1)
            mortality_by_freq = measurement_counts.groupby(self.mortality_data).mean()
            freq_summary[var] = mortality_by_freq
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.mortality_data, y=measurement_counts)
            plt.title(f"Measurement Frequency vs Mortality for {var}")
            plt.xlabel("Mortality")
            plt.ylabel("Measurement Frequency")
            plt.show()
            print(f"Measurement Frequency Summary for {var}:\n{mortality_by_freq}\n")
        return freq_summary

    def run_all(self):
        print("NaN Summary:")
        nan_summary = self.nan_summary()

        print("\nNaN Heatmap:")
        self.plot_nan_heatmap()

        print("\nNaN Distribution:")
        self.plot_nan_distribution()

        print("\nNaN Distribution by Mortality:")
        self.plot_nan_distribution_by_mortality()

        print("\nStatistical Summary:")
        stats_summary = self.statistical_summary()

        print("\nTime Series Plots:")
        self.plot_time_series()

        print("\nHistograms:")
        self.plot_histograms()

        print("\nMeasurement Frequency Analysis:")
        freq_analysis = self.measurement_frequency_analysis()
