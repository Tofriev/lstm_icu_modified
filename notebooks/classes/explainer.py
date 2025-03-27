import shap
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import json


class SHAPExplainer:
    def __init__(self, model, feature_names=None, dual_shap_plot=True):
        self.orig_model = model 
        self.model = model
        self.shap_values = None
        self.test_seq_data_np = None   # (num_samples, time_steps, n_seq_features)
        self.test_static_data_np = None  # (num_samples, n_static_features)
        self.scale_factors = None
        self.feature_names = feature_names

        self.dual_shap_plot = dual_shap_plot

    def pad_sequences(self, sequences, max_len=None):
        """
        Pad a list of 2D arrays (time_steps, n_features) with zeros so all have the same number of time steps.
        """
        if max_len is None:
            max_len = max(seq.shape[0] for seq in sequences)
        n_features = sequences[0].shape[1]
        padded = np.zeros((len(sequences), max_len, n_features), dtype=np.float32)
        for i, seq in enumerate(sequences):
            length = seq.shape[0]
            padded[i, :length, :] = seq
        return padded

    def extract_shap_values(self, sequences, num_samples, batch_size=10, background_pct=0.1, random_seed=42):
       
        print("Extracting SHAP values...")

        # sequential and static parts
        seq_list = [s[0] for s in sequences]
        all_seq_data_np = self.pad_sequences(seq_list)  # shape: (total_samples, max_time_steps, n_seq_features)
        all_static_data_np = np.array([s[1] for s in sequences])  # shape: (total_samples, n_static_features)
        all_labels_np = np.array([s[2] for s in sequences]) 
        total_samples = all_seq_data_np.shape[0]
        
        # print(f"Sequential data shape (after padding): {all_seq_data_np.shape}")
        # print(f"Static data shape: {all_static_data_np.shape}")
        # print("First sequential sample:\n", all_seq_data_np[0])
        # print("First static sample:\n", all_static_data_np[0])
        

        np.random.seed(random_seed)
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        num_background = int(background_pct * total_samples)
        background_idx = indices[:num_background]
        test_idx = indices[num_background:]
        
        # print(f"N background samples: {num_background}")
        # print(f"N test samples before limiting: {len(test_idx)}")
        
        background_seq_np = all_seq_data_np[background_idx]
        background_static_np = all_static_data_np[background_idx]
        test_seq_np = all_seq_data_np[test_idx]
        test_static_np = all_static_data_np[test_idx]
        test_labels_np = all_labels_np[test_idx]
        self.test_labels_np = test_labels_np
        
        if num_samples > len(test_seq_np):
            num_samples = len(test_seq_np)
        test_seq_np = test_seq_np[:num_samples]
        test_static_np = test_static_np[:num_samples]
        
        

        self.test_seq_data_np = test_seq_np
        self.test_static_data_np = test_static_np
        
        # print(f"Background sequential shape: {background_seq_np.shape}")
        # print(f"Background static shape: {background_static_np.shape}")
        # print(f"Test sequential shape: {test_seq_np.shape}")
        # print(f"Test static shape: {test_static_np.shape}")

        background_seq_tensor = torch.tensor(background_seq_np).float()
        background_static_tensor = torch.tensor(background_static_np).float()
        explainer = shap.DeepExplainer(self.model, [background_seq_tensor, background_static_tensor])
        
        shap_values_batches = []
        for i in tqdm(range(0, num_samples, batch_size), desc="Processing Batches"):
            batch_seq = torch.tensor(test_seq_np[i : i + batch_size]).float()
            batch_static = torch.tensor(test_static_np[i : i + batch_size]).float()
           
            shap_values_batch = explainer.shap_values([batch_seq, batch_static])
            shap_values_batches.append(shap_values_batch)
            #print(f"Processed batch {i // batch_size + 1}")
    
        # [ [shap_seq_class0, shap_static_class0], [shap_seq_class1, shap_static_class1] ]
        aggregated_shap_values = []
        num_outputs = len(shap_values_batches[0])  # number of class outputs
        for out in range(num_outputs):
            aggregated_for_output = []
            num_branches = len(shap_values_batches[0][out])  
            for branch in range(num_branches):
                arrays = [np.array(batch[out][branch]) for batch in shap_values_batches]
                if arrays[0].ndim == 3:
                    max_time = max(arr.shape[1] for arr in arrays)
                    padded_arrays = []
                    for arr in arrays:
                        if arr.shape[1] < max_time:
                            pad_width = ((0, 0), (0, max_time - arr.shape[1]), (0, 0))
                            arr_padded = np.pad(arr, pad_width, mode='constant')
                            padded_arrays.append(arr_padded)
                        else:
                            padded_arrays.append(arr)
                    aggregated = np.concatenate(padded_arrays, axis=0)
                elif arrays[0].ndim == 2:
                    aggregated = np.concatenate(arrays, axis=0)
                else:
                    raise ValueError("Unexpected SHAP array dimension.")
                aggregated_for_output.append(aggregated)
            aggregated_shap_values.append(aggregated_for_output)
        self.shap_values = aggregated_shap_values

    def explain_with_ordinary_SHAP(self, feature_names):
        """
        Visualize the SHAP summary plot using the sequential input.
        """
        #print("Visualizing ordinary SHAP summary plot...")
        num_features = self.test_seq_data_np.shape[2]
        aggregated_shap_values = self.shap_values[1][0].mean(axis=1)   # (num_samples, n_seq_features)
        aggregated_test_data = self.test_seq_data_np.mean(axis=1)      # (num_samples, n_seq_features)
        
        shap.summary_plot(
            aggregated_shap_values,
            aggregated_test_data,
            feature_names or [f"Feature {i}" for i in range(num_features)],
            show=True,
        )
    
    def plot_shap_heatmap_feature_rank(self, feature_names):
        #("Plotting SHAP heatmap (mean feature rank) for all inputs...")
        if self.shap_values is None:
            raise ValueError("Run extract_shap_values first.")
        
        if self.dual_shap_plot:
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))
            for idx, (class_idx, cmap, title_label) in enumerate([(1, "Reds_r", "Non Survival"), (0, "Blues_r", "Survival")]):
                shap_seq = self.shap_values[class_idx][0]      # sequential branch
                shap_static = self.shap_values[class_idx][1]     # static branch
                ts_length = self.test_seq_data_np.shape[1] if self.test_seq_data_np is not None else 10
                shap_static_repl = self.replicate_static(shap_static, ts_length)
                shap_combined = np.concatenate([shap_seq, shap_static_repl], axis=2)
                
                shap_abs = np.abs(shap_combined)
                ranks = np.argsort(np.argsort(-shap_abs, axis=2), axis=2) + 1
                mean_ranks = ranks.mean(axis=0)
                overall_mean = mean_ranks.mean(axis=0)
                
                sorted_indices = np.argsort(overall_mean)
                sorted_feature_names = [feature_names[i] for i in sorted_indices]
                sorted_mean_ranks = mean_ranks[:, sorted_indices].T
                
                time_steps = sorted_mean_ranks.shape[1]
                df = pd.DataFrame(
                    sorted_mean_ranks,
                    index=sorted_feature_names,
                    columns=[f"Time {i}" for i in range(time_steps)]
                )
                sns.heatmap(df, cmap=cmap, annot=False, fmt=".2f", cbar_kws={"label": "Mean Rank"}, ax=axs[idx])
                axs[idx].set_title(f"SHAP Heatmap (Mean Feature Rank) - {title_label}")
                axs[idx].set_xlabel("Time Steps")
                axs[idx].set_ylabel("Features (Sorted by Overall Importance)")
            plt.tight_layout()
            plt.show()
        else:
            shap_seq = self.shap_values[1][0]
            shap_static = self.shap_values[1][1]
            ts_length = self.test_seq_data_np.shape[1] if self.test_seq_data_np is not None else 10
            shap_static_repl = self.replicate_static(shap_static, ts_length)
            shap_combined = np.concatenate([shap_seq, shap_static_repl], axis=2)
            
            shap_abs = np.abs(shap_combined)
            ranks = np.argsort(np.argsort(-shap_abs, axis=2), axis=2) + 1
            mean_ranks = ranks.mean(axis=0)
            overall_mean = mean_ranks.mean(axis=0)
            
            sorted_indices = np.argsort(overall_mean)
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            sorted_mean_ranks = mean_ranks[:, sorted_indices].T
            
            time_steps = sorted_mean_ranks.shape[1]
            df = pd.DataFrame(
                sorted_mean_ranks,
                index=sorted_feature_names,
                columns=[f"Time {i}" for i in range(time_steps)]
            )
            plt.figure(figsize=(10, 8))
            sns.heatmap(df, cmap="Reds_r", annot=False, fmt=".2f", cbar_kws={"label": "Mean Rank"})
            plt.title("SHAP Heatmap (Mean Feature Rank) for Non Survival")
            plt.xlabel("Time Steps")
            plt.ylabel("Features (Sorted by Overall Importance)")
            plt.show()

    def plot_shap_heatmap_mean_abs(self, feature_names):
        #print("Plotting SHAP heatmap (mean absolute SHAP) for all inputs...")
        if self.shap_values is None:
            raise ValueError("Run extract_shap_values first.")

        if self.dual_shap_plot:
            fig, axs = plt.subplots(1, 2, figsize=(20, 8))

            for idx, (class_idx, cmap, title_label) in enumerate(
                [(1, "Reds", "Non Survival"), (0, "Blues", "Survival")]
            ):
                shap_seq = self.shap_values[class_idx][0]
                shap_static = self.shap_values[class_idx][1]
                ts_length = self.test_seq_data_np.shape[1] if self.test_seq_data_np is not None else 10
                shap_static_repl = self.replicate_static(shap_static, ts_length)
                
                shap_combined = np.concatenate([shap_seq, shap_static_repl], axis=2)
                
                shap_abs = np.abs(shap_combined)
                mean_abs = shap_abs.mean(axis=0)  # shape: (time_steps, total_features)
                overall_mean = mean_abs.mean(axis=0)  # shape: (total_features,)

                sorted_indices = np.argsort(overall_mean)[::-1]
                sorted_feature_names = [feature_names[i] for i in sorted_indices]

                sorted_mean_abs = mean_abs[:, sorted_indices].T  # shape: (total_features, time_steps)

                time_steps = sorted_mean_abs.shape[1]
                df = pd.DataFrame(
                    sorted_mean_abs,
                    index=sorted_feature_names,
                    columns=[f"Time {i}" for i in range(time_steps)]
                )

                df_clipped = df.clip(lower=1e-6)

                sns.heatmap(
                    df_clipped,
                    cmap=cmap,
                    annot=False,
                    fmt=".2f",
                    ax=axs[idx],
                    cbar_kws={"label": "Mean Absolute SHAP Score (Log Scale)"},
                    norm=LogNorm(vmin=df_clipped.min().min(), vmax=df_clipped.max().max()),
                )

                axs[idx].set_title(f"SHAP Heatmap (Mean Absolute SHAP) - {title_label} [Log Scale]")
                axs[idx].set_xlabel("Time Steps")
                axs[idx].set_ylabel("Features (Sorted by Overall Importance)")

            plt.tight_layout()
            plt.show()

        else:
            shap_seq = self.shap_values[1][0]
            shap_static = self.shap_values[1][1]
            ts_length = self.test_seq_data_np.shape[1] if self.test_seq_data_np is not None else 10
            shap_static_repl = self.replicate_static(shap_static, ts_length)
            
            shap_combined = np.concatenate([shap_seq, shap_static_repl], axis=2)
            
            shap_abs = np.abs(shap_combined)
            mean_abs = shap_abs.mean(axis=0)
            overall_mean = mean_abs.mean(axis=0)
            
            sorted_indices = np.argsort(overall_mean)[::-1]
            sorted_feature_names = [feature_names[i] for i in sorted_indices]
            sorted_mean_abs = mean_abs[:, sorted_indices].T
            
            time_steps = sorted_mean_abs.shape[1]
            df = pd.DataFrame(
                sorted_mean_abs,
                index=sorted_feature_names,
                columns=[f"Time {i}" for i in range(time_steps)]
            )

            df_clipped = df.clip(lower=1e-6)

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                df_clipped,
                cmap="Reds",
                annot=False,
                fmt=".2f",
                cbar_kws={"label": "Mean Absolute SHAP Score (Log Scale)"},
                norm=LogNorm(vmin=df_clipped.min().min(), vmax=df_clipped.max().max()),
            )
            plt.title("SHAP Heatmap (Mean Absolute SHAP) for Non Survival [Log Scale]")
            plt.xlabel("Time Steps")
            plt.ylabel("Features (Sorted by Overall Importance)")
            plt.tight_layout()
            plt.show()

    def plot_single_feature_time_shap(self, sample_idx, variable_name, scaler=None, input_type=None, feature_idx=None):
        if input_type == 'sequential':
            branch_idx = 0
            data_array = self.test_seq_data_np
            ts_length = data_array.shape[1]
            raw_values = data_array[sample_idx, :, feature_idx]
        elif input_type == 'static':
            branch_idx = 1
            data_array = self.test_static_data_np
            ts_length = self.test_seq_data_np.shape[1] if self.test_seq_data_np is not None else 10
            raw_value = data_array[sample_idx, feature_idx]
            raw_values = np.repeat(raw_value, ts_length)
        else:
            raise ValueError("input_type must be either 'sequential' or 'static'.")

        if scaler is not None:
            feature_scale = scaler.scale_[feature_idx]
            feature_mean = scaler.mean_[feature_idx]
            feature_values = raw_values * feature_scale + feature_mean
        else:
            feature_values = raw_values

        time_steps = np.arange(ts_length)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Feature Value", color="black")
        ax1.plot(time_steps, feature_values, color="black", linewidth=2.5, label="Feature Value")
        ax1.tick_params(axis="y", labelcolor="black")

        ax2 = ax1.twinx()
        ax2.set_ylabel("SHAP Value", color="black")
        shap_all_values = []

        if self.dual_shap_plot:
            if input_type == 'sequential':
                shap_vals_non_surv = self.shap_values[1][branch_idx][sample_idx, :, feature_idx]
                shap_vals_surv = self.shap_values[0][branch_idx][sample_idx, :, feature_idx]
            else:
                shap_non = self.shap_values[1][branch_idx][sample_idx, feature_idx]
                shap_surv = self.shap_values[0][branch_idx][sample_idx, feature_idx]
                shap_vals_non_surv = np.repeat(shap_non, ts_length)
                shap_vals_surv = np.repeat(shap_surv, ts_length)

            ax2.plot(time_steps, shap_vals_non_surv, color="tab:red", linestyle="--", linewidth=1.5,
                    alpha=0.6, label="SHAP (Non Survival)")
            ax2.plot(time_steps, shap_vals_surv, color="tab:blue", linestyle="--", linewidth=1.5,
                    alpha=0.6, label="SHAP (Survival)")
            shap_all_values.extend(shap_vals_non_surv)
            shap_all_values.extend(shap_vals_surv)

        else:
            if input_type == 'sequential':
                shap_vals = self.shap_values[1][branch_idx][sample_idx, :, feature_idx]
            else:
                shap_val = self.shap_values[1][branch_idx][sample_idx, feature_idx]
                shap_vals = np.repeat(shap_val, ts_length)

            ax2.plot(time_steps, shap_vals, color="tab:red", linestyle="--", linewidth=1.5,
                    alpha=0.6, label="SHAP (Non Survival)")
            shap_all_values.extend(shap_vals)

        if len(shap_all_values) > 0:
            min_val = np.min(shap_all_values)
            max_val = np.max(shap_all_values)
            margin = 0.1 * (max_val - min_val) if (max_val - min_val) != 0 else 0.1
            ax2.set_ylim(min_val - margin, max_val + margin)
        ax2.tick_params(axis="y", labelcolor="black")


        if input_type == 'static':
            smin, smax = ax2.get_ylim()
            shift = 0.05 * (smax - smin)
            ax2.set_ylim(smin + shift, smax + shift)

        plt.title(f"Feature: {variable_name} (Sample {sample_idx})")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")


        plt.subplots_adjust(bottom=0.16)  

        actual_label_val = self.test_labels_np[sample_idx]
        actual_label_str = "Survival" if actual_label_val == 0 else "Non Survival"

        self.orig_model.eval()
        with torch.no_grad():
            sample_seq = self.test_seq_data_np[sample_idx : sample_idx+1]
            sample_static = self.test_static_data_np[sample_idx : sample_idx+1]
            seq_tensor = torch.tensor(sample_seq).float()
            static_tensor = torch.tensor(sample_static).float()
            out = self.orig_model(seq_tensor, static_tensor)
            out = out.squeeze()  

            p_non_surv = None
            predicted_class_str = None
            if out.dim() == 0:
                p_non_surv = float(out.item())
                p_non_surv = max(min(p_non_surv, 1.0), 0.0)
                p_surv = 1.0 - p_non_surv
                predicted_class_str = "Non Survival" if p_non_surv >= 0.5 else "Survival"
            elif out.shape[0] == 2:
                probs = torch.softmax(out, dim=0)
                p_surv = probs[0].item()
                p_non_surv = probs[1].item()
                predicted_class_str = "Survival" if p_surv >= p_non_surv else "Non Survival"
                predicted_class_p = p_surv if p_surv >= p_non_surv else p_non_surv


        text_str = (
            f"Predicted: {predicted_class_str} "
            f"(P={predicted_class_p:.2f}) "
            f"Actual: {actual_label_str}"
        )

        fig.text(
            0.5, 0.02,             
            text_str,
            ha='center',
            va='bottom',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5')
        )

        plt.show()





    
    def is_probability_output(self, sample_input):
        """
        Check if model output is a probability.
        sample_input should be a tuple: (seq_tensor, static_tensor)
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(*sample_input)
            #print(f"Model output: {output}")
            output_np = output.cpu().numpy().flatten()
            return np.all((output_np >= 0.0) & (output_np <= 1.0))
    
    def explain(self, sequences, method, num_samples=1000, scaler=None, numerical_features=None,
                feature_to_explain=None,  feature_type=None, feature_idx=None, batch_size=10, save_shap_values=True):
        """
        Main method to extract and visualize SHAP values.
        sequences: list of tuples (sequential_features, static_features, label)
        """
        sample_np_seq = np.array([sequences[0][0]])
        sample_np_static = np.array([sequences[0][1]])
        sample_torch = (torch.tensor(sample_np_seq).float(), torch.tensor(sample_np_static).float())
        
        if self.is_probability_output(sample_torch):
           # print("Model outputs probabilities: wrapping to logits...")
            self.model = LogitWrapper(self.model)
        # else:
        #     print("Model outputs logits.")
        
        self.extract_shap_values(sequences, num_samples, batch_size)
        if save_shap_values:
            self.save_shap_values("shap_values.json")
       # print("Feature names:", self.feature_names)
        if method == "ordinary_SHAP":
            self.explain_with_ordinary_SHAP(self.feature_names)
        elif method == "heatmap_SHAP":
            self.plot_shap_heatmap_mean_abs(self.feature_names)
        elif method == "feature_rank_heatmap_SHAP":
            self.plot_shap_heatmap_feature_rank(self.feature_names)
        elif method == "plot_single_feature_time_shap":
            self.plot_single_feature_time_shap(sample_idx=1, variable_name=feature_to_explain,
                                                 scaler=scaler, input_type=feature_type, feature_idx=feature_idx)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def replicate_static(self, shap_array, time_steps):
        # shap_array shape: (num_samples, n_features)
        return np.repeat(shap_array[:, np.newaxis, :], time_steps, axis=1)
    
    def save_shap_values(self, save_path):
        """
        Save the aggregated SHAP values into a JSON file.
        The structure is organized by class label and branch (sequential/static).
        """
        data_to_save = {
            "label_0": {  # Survival 
                "sequential": self.shap_values[0][0].tolist(),
                "static": self.shap_values[0][1].tolist()
            },
            "label_1": {  # Non Survival 
                "sequential": self.shap_values[1][0].tolist(),
                "static": self.shap_values[1][1].tolist()
            }
        }
        with open(save_path, "w") as f:
            json.dump(data_to_save, f, indent=4)
        #print(f"SHAP values saved to {save_path}")

    

class LogitWrapper(torch.nn.Module):
    """
    Wraps a model that returns probabilities so that it returns logits.
    Assumes a binary classification scenario.
    """
    def __init__(self, model):
        super(LogitWrapper, self).__init__()
        self.model = model

    def forward(self, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            args = args[0]
        p = self.model(*args)
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        logits = torch.log(p / (1 - p))
        return logits
