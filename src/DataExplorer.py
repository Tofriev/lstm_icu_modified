import matplotlib.pyplot as plt
import seaborn as sns
import random


class DataExplorer:
    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset.data is None:
            self.dataset.load_data()

    def exploratory_data_analysis(self):
        """Perform exploratory data analysis to show the distribution of mortality."""
        mortality_counts = self.dataset.data.drop_duplicates(subset=["stay_id"])[
            "mortality"
        ].value_counts()

        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=mortality_counts.index, y=mortality_counts.values, palette="viridis"
        )
        plt.title("Distribution of Mortality")
        plt.xlabel("Mortality (0: Not Died, 1: Died)")
        plt.ylabel("Count")
        plt.show()

        # Distribution of respiratory rates
        plt.figure(figsize=(10, 6))
        sns.histplot(self.dataset.data["rr_mean"], bins=30, kde=True)
        plt.title("Distribution of Respiratory Rates")
        plt.xlabel("Respiratory Rate")
        plt.ylabel("Frequency")
        plt.show()

        # Mortality rate over time
        self.dataset.data["hour"] = self.dataset.data["hour"].astype(int)
        hourly_mortality = self.dataset.data.groupby("hour")["mortality"].mean()

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=hourly_mortality.index, y=hourly_mortality.values)
        plt.title("Mortality Rate Over Time (First 24 Hours)")
        plt.xlabel("Hour")
        plt.ylabel("Mortality Rate")
        plt.show()

        # Box plot of respiratory rate by mortality status
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            x="mortality", y="rr_mean", data=self.dataset.data, palette="viridis"
        )
        plt.title("Respiratory Rate by Mortality Status")
        plt.xlabel("Mortality (0: Not Died, 1: Died)")
        plt.ylabel("Respiratory Rate")
        plt.show()

    def plot_patient_chart(self, stay_id=None):
        """Plot the respiratory rate chart for a specific patient or a random patient if stay_id is not provided."""
        if stay_id is None:
            stay_id = random.choice(self.dataset.data["stay_id"].unique())

        patient_data = self.dataset.data[self.dataset.data["stay_id"] == stay_id]
        if patient_data.empty:
            print(f"No data found for stay_id: {stay_id}")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(patient_data["hour"], patient_data["rr_mean"], marker="o")
        plt.title(f"Respiratory Rate Chart for Patient with stay_id: {stay_id}")
        plt.xlabel("Hour")
        plt.ylabel("Respiratory Rate")
        plt.grid(True)
        plt.show()
